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
def alreadyParsedRule(self, input, ruleIndex):
    """
        Has this rule already parsed input at the current index in the
        input stream?  Return the stop token index or MEMO_RULE_UNKNOWN.
        If we attempted but failed to parse properly before, return
        MEMO_RULE_FAILED.

        This method has a side-effect: if we have seen this input for
        this rule and successfully parsed before, then seek ahead to
        1 past the stop token matched for this rule last time.
        """
    stopIndex = self.getRuleMemoization(ruleIndex, input.index())
    if stopIndex == self.MEMO_RULE_UNKNOWN:
        return False
    if stopIndex == self.MEMO_RULE_FAILED:
        raise BacktrackingFailed
    else:
        input.seek(stopIndex + 1)
    return True