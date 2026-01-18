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
def open_scope(self, node, single_paren=False):
    """Open a parenthesized scope on the given node."""
    result = ''
    parens = []
    start_i = self._i
    start_loc = prev_loc = self._loc
    for tok in self.takewhile(lambda t: t.type in FORMATTING_TOKENS or t.src == '('):
        result += self._space_between(prev_loc, tok.start)
        if tok.src == '(' and single_paren and parens:
            self.rewind()
            self._loc = tok.start
            break
        result += tok.src
        if tok.src == '(':
            parens.append(result)
            result = ''
            start_i = self._i
            start_loc = self._loc
        prev_loc = self._loc
    if parens:
        next_tok = self.peek()
        parens[-1] += result + self._space_between(self._loc, next_tok.start)
        self._loc = next_tok.start
        for paren in parens:
            self._parens.append(paren)
            self._scope_stack.append(_scope_helper(node))
    else:
        self._i = start_i
        self._loc = start_loc