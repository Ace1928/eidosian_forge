from antlr4 import *
from io import StringIO
import sys
def errorCapturingIdentifierExtra(self):
    localctx = fugue_sqlParser.ErrorCapturingIdentifierExtraContext(self, self._ctx, self.state)
    self.enterRule(localctx, 412, self.RULE_errorCapturingIdentifierExtra)
    self._la = 0
    try:
        localctx = fugue_sqlParser.ErrorIdentContext(self, localctx)
        self.enterOuterAlt(localctx, 1)
        self.state = 3803
        self._errHandler.sync(self)
        _la = self._input.LA(1)
        while True:
            self.state = 3801
            self.match(fugue_sqlParser.MINUS)
            self.state = 3802
            self.identifier()
            self.state = 3805
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if not _la == 320:
                break
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx