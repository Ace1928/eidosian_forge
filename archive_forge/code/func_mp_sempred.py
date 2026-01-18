from antlr4 import *
from io import StringIO
import sys
def mp_sempred(self, localctx: MpContext, predIndex: int):
    if predIndex == 2:
        return self.precpred(self._ctx, 2)