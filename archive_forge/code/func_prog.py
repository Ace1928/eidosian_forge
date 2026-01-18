from antlr4 import *
from io import StringIO
import sys
def prog(self):
    localctx = AutolevParser.ProgContext(self, self._ctx, self.state)
    self.enterRule(localctx, 0, self.RULE_prog)
    self._la = 0
    try:
        self.enterOuterAlt(localctx, 1)
        self.state = 57
        self._errHandler.sync(self)
        _la = self._input.LA(1)
        while True:
            self.state = 56
            self.stat()
            self.state = 59
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if not (_la & ~63 == 0 and 1 << _la & 299067041120256 != 0):
                break
    except RecognitionException as re:
        localctx.exception = re
        self._errHandler.reportError(self, re)
        self._errHandler.recover(self, re)
    finally:
        self.exitRule()
    return localctx