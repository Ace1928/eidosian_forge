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
def matchRange(self, a, b):
    if self.input.LA(1) < a or self.input.LA(1) > b:
        if self._state.backtracking > 0:
            raise BacktrackingFailed
        mre = MismatchedRangeException(unichr(a), unichr(b), self.input)
        self.recover(mre)
        raise mre
    self.input.consume()