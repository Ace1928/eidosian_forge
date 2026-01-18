from antlr4 import *
from io import StringIO
import sys
def supeq(self):
    localctx = LaTeXParser.SupeqContext(self, self._ctx, self.state)
    self.enterRule(localctx, 80, self.RULE_supeq)
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 516
        self.match(LaTeXParser.UNDERSCORE)
        self.state = 517
        self.match(LaTeXParser.L_BRACE)
        self.state = 518
        self.equality()
        self.state = 519
        self.match(LaTeXParser.R_BRACE)
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx