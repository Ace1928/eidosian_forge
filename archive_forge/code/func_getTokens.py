from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def getTokens(self, start=None, stop=None, types=None):
    """
        Given a start and stop index, return a list of all tokens in
        the token type set.  Return None if no tokens were found.  This
        method looks at both on and off channel tokens.
        """
    if self.p == -1:
        self.fillBuffer()
    if stop is None or stop >= len(self.tokens):
        stop = len(self.tokens) - 1
    if start is None or stop < 0:
        start = 0
    if start > stop:
        return None
    if isinstance(types, six.integer_types):
        types = set([types])
    filteredTokens = [token for token in self.tokens[start:stop] if types is None or token.type in types]
    if len(filteredTokens) == 0:
        return None
    return filteredTokens