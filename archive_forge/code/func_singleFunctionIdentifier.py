from antlr4 import *
from io import StringIO
import sys
def singleFunctionIdentifier(self):
    localctx = fugue_sqlParser.SingleFunctionIdentifierContext(self, self._ctx, self.state)
    self.enterRule(localctx, 160, self.RULE_singleFunctionIdentifier)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1163
        self.functionIdentifier()
        self.state = 1164
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx