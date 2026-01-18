from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
def getMissingSymbol(self, input, e, expectedTokenType, follow):
    if expectedTokenType == EOF:
        tokenText = '<missing EOF>'
    else:
        tokenText = '<missing ' + self.tokenNames[expectedTokenType] + '>'
    t = CommonToken(type=expectedTokenType, text=tokenText)
    current = input.LT(1)
    if current.type == EOF:
        current = input.LT(-1)
    if current is not None:
        t.line = current.line
        t.charPositionInLine = current.charPositionInLine
    t.channel = DEFAULT_CHANNEL
    return t