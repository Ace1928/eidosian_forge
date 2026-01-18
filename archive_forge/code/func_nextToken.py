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
def nextToken(self):
    """
        Return a token from this source; i.e., match a token on the char
        stream.
        """
    while 1:
        self._state.token = None
        self._state.channel = DEFAULT_CHANNEL
        self._state.tokenStartCharIndex = self.input.index()
        self._state.tokenStartCharPositionInLine = self.input.charPositionInLine
        self._state.tokenStartLine = self.input.line
        self._state.text = None
        if self.input.LA(1) == EOF:
            return EOF_TOKEN
        try:
            self.mTokens()
            if self._state.token is None:
                self.emit()
            elif self._state.token == SKIP_TOKEN:
                continue
            return self._state.token
        except NoViableAltException as re:
            self.reportError(re)
            self.recover(re)
        except RecognitionException as re:
            self.reportError(re)