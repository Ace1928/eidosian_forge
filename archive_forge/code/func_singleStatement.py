from antlr4 import *
from io import StringIO
import sys
def singleStatement(self):
    localctx = fugue_sqlParser.SingleStatementContext(self, self._ctx, self.state)
    self.enterRule(localctx, 152, self.RULE_singleStatement)
    self._la = 0
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1145
        self.statement()
        self.state = 1149
        self._errHandler.sync(self)
        _la = self._input.LA(1)
        while _la == 13:
            self.state = 1146
            self.match(fugue_sqlParser.T__12)
            self.state = 1151
            self._errHandler.sync(self)
            _la = self._input.LA(1)
        self.state = 1152
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx