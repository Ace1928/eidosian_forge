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
def getText(self):
    """
        Return the text matched so far for the current token or any
        text override.
        """
    if self._state.text is not None:
        return self._state.text
    return self.input.substring(self._state.tokenStartCharIndex, self.getCharIndex() - 1)