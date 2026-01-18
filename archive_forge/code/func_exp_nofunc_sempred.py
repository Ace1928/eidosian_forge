from antlr4 import *
from io import StringIO
import sys
def exp_nofunc_sempred(self, localctx: Exp_nofuncContext, predIndex: int):
    if predIndex == 5:
        return self.precpred(self._ctx, 2)