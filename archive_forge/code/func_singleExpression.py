from antlr4 import *
from io import StringIO
import sys
def singleExpression(self):
    localctx = fugue_sqlParser.SingleExpressionContext(self, self._ctx, self.state)
    self.enterRule(localctx, 154, self.RULE_singleExpression)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1154
        self.namedExpression()
        self.state = 1155
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx