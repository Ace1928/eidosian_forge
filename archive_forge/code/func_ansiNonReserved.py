from antlr4 import *
from io import StringIO
import sys
def ansiNonReserved(self):
    localctx = fugue_sqlParser.AnsiNonReservedContext(self, self._ctx, self.state)
    self.enterRule(localctx, 424, self.RULE_ansiNonReserved)
    self._la = 0
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 3865
        _la = self._input.LA(1)
        if not (_la - 58 & ~63 == 0 and 1 << _la - 58 & -4616724533169136869 != 0 or (_la - 122 & ~63 == 0 and 1 << _la - 122 & -54836095400108079 != 0) or (_la - 186 & ~63 == 0 and 1 << _la - 186 & -72339344050251969 != 0) or (_la - 250 & ~63 == 0 and 1 << _la - 250 & 176704157053345137 != 0) or (_la == 324)):
            self._errHandler.recoverInline(self)
        else:
            self._errHandler.reportMatch(self)
            self.consume()
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx