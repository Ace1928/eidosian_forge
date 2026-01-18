from antlr4 import *
from io import StringIO
import sys
def queryTerm_sempred(self, localctx: QueryTermContext, predIndex: int):
    if predIndex == 1:
        return self.precpred(self._ctx, 3)
    if predIndex == 2:
        return self.precpred(self._ctx, 2)
    if predIndex == 3:
        return self.precpred(self._ctx, 1)