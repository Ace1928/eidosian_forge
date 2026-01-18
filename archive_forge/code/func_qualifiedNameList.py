from antlr4 import *
from io import StringIO
import sys
def qualifiedNameList(self):
    localctx = fugue_sqlParser.QualifiedNameListContext(self, self._ctx, self.state)
    self.enterRule(localctx, 404, self.RULE_qualifiedNameList)
    self._la = 0
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 3777
        self.qualifiedName()
        self.state = 3782
        self._errHandler.sync(self)
        _la = self._input.LA(1)
        while _la == 2:
            self.state = 3778
            self.match(fugue_sqlParser.T__1)
            self.state = 3779
            self.qualifiedName()
            self.state = 3784
            self._errHandler.sync(self)
            _la = self._input.LA(1)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx