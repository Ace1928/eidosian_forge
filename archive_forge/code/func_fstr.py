from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def fstr(self):
    """Parses an fstring, including subexpressions.

    Returns:
      A generator function which, when repeatedly reads a chunk of the fstring
      up until the next subexpression and yields that chunk, plus a new token
      generator to use to parse the subexpression. The subexpressions in the
      original fstring data are replaced by placeholders to make it possible to
      fill them in with new values, if desired.
    """

    def fstr_parser():
        if self.peek_non_whitespace().type == TOKENS.STRING:
            str_content = self.str()
        else:

            def fstr_eater(tok):
                if tok.type == TOKENS.OP and tok.src == '}':
                    if fstr_eater.level <= 0:
                        return False
                    fstr_eater.level -= 1
                if tok.type == TOKENS.OP and tok.src == '{':
                    fstr_eater.level += 1
                return True
            fstr_eater.level = 0
            str_content = self.eat_tokens(fstr_eater)
        indexed_chars = enumerate(str_content)
        val_idx = 0
        i = -1
        result = ''
        while i < len(str_content) - 1:
            i, c = next(indexed_chars)
            result += c
            if c == '{':
                nexti, nextc = next(indexed_chars)
                if nextc == '{':
                    result += c
                    continue
                indexed_chars = itertools.chain([(nexti, nextc)], indexed_chars)
                result += fstring_utils.placeholder(val_idx) + '}'
                val_idx += 1
                tg = TokenGenerator(str_content[i + 1:], ignore_error_token=True)
                yield (result, tg)
                result = ''
                for tg_i in range(tg.chars_consumed()):
                    i, c = next(indexed_chars)
                i, c = next(indexed_chars)
                while c != '}':
                    i, c = next(indexed_chars)
        yield (result, None)
    return fstr_parser