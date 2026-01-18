from antlr4 import *
from io import StringIO
import sys
def fugueJson(self):
    localctx = fugue_sqlParser.FugueJsonContext(self, self._ctx, self.state)
    self.enterRule(localctx, 128, self.RULE_fugueJson)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1071
        self.fugueJsonValue()
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx