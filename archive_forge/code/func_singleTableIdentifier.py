from antlr4 import *
from io import StringIO
import sys
def singleTableIdentifier(self):
    localctx = fugue_sqlParser.SingleTableIdentifierContext(self, self._ctx, self.state)
    self.enterRule(localctx, 156, self.RULE_singleTableIdentifier)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1157
        self.tableIdentifier()
        self.state = 1158
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx