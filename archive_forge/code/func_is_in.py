import re
from bisect import bisect
from pygments.lexer import RegexLexer, include, default, bygroups, using, this
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
def is_in(self, w, mapping):
    """
        It's kind of difficult to decide if something might be a keyword
        in VimL because it allows you to abbreviate them.  In fact,
        'ab[breviate]' is a good example.  :ab, :abbre, or :abbreviate are
        valid ways to call it so rather than making really awful regexps
        like::

            \\bab(?:b(?:r(?:e(?:v(?:i(?:a(?:t(?:e)?)?)?)?)?)?)?)?\\b

        we match `\\b\\w+\\b` and then call is_in() on those tokens.  See
        `scripts/get_vimkw.py` for how the lists are extracted.
        """
    p = bisect(mapping, (w,))
    if p > 0:
        if mapping[p - 1][0] == w[:len(mapping[p - 1][0])] and mapping[p - 1][1][:len(w)] == w:
            return True
    if p < len(mapping):
        return mapping[p][0] == w[:len(mapping[p][0])] and mapping[p][1][:len(w)] == w
    return False