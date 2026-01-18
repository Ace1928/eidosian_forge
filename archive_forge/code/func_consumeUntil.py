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
def consumeUntil(self, input, tokenTypes):
    """
        Consume tokens until one matches the given token or token set

        tokenTypes can be a single token type or a set of token types

        """
    if not isinstance(tokenTypes, (set, frozenset)):
        tokenTypes = frozenset([tokenTypes])
    ttype = input.LA(1)
    while ttype != EOF and ttype not in tokenTypes:
        input.consume()
        ttype = input.LA(1)