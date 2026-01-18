from antlr4 import *
from io import StringIO
import sys
def singleMultipartIdentifier(self):
    localctx = fugue_sqlParser.SingleMultipartIdentifierContext(self, self._ctx, self.state)
    self.enterRule(localctx, 158, self.RULE_singleMultipartIdentifier)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 1160
        self.multipartIdentifier()
        self.state = 1161
        self.match(fugue_sqlParser.EOF)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx