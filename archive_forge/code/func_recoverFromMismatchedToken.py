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
def recoverFromMismatchedToken(self, input, ttype, follow):
    """Attempt to recover from a single missing or extra token.

        EXTRA TOKEN

        LA(1) is not what we are looking for.  If LA(2) has the right token,
        however, then assume LA(1) is some extra spurious token.  Delete it
        and LA(2) as if we were doing a normal match(), which advances the
        input.

        MISSING TOKEN

        If current token is consistent with what could come after
        ttype then it is ok to 'insert' the missing token, else throw
        exception For example, Input 'i=(3;' is clearly missing the
        ')'.  When the parser returns from the nested call to expr, it
        will have call chain:

          stat -> expr -> atom

        and it will be trying to match the ')' at this point in the
        derivation:

             => ID '=' '(' INT ')' ('+' atom)* ';'
                                ^
        match() will see that ';' doesn't match ')' and report a
        mismatched token error.  To recover, it sees that LA(1)==';'
        is in the set of tokens that can follow the ')' token
        reference in rule atom.  It can assume that you forgot the ')'.
        """
    e = None
    if self.mismatchIsUnwantedToken(input, ttype):
        e = UnwantedTokenException(ttype, input)
        self.beginResync()
        input.consume()
        self.endResync()
        self.reportError(e)
        matchedSymbol = self.getCurrentInputSymbol(input)
        input.consume()
        return matchedSymbol
    if self.mismatchIsMissingToken(input, follow):
        inserted = self.getMissingSymbol(input, e, ttype, follow)
        e = MissingTokenException(ttype, input, inserted)
        self.reportError(e)
        return inserted
    e = MismatchedTokenException(ttype, input)
    raise e