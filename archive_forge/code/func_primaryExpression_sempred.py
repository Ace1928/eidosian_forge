from antlr4 import *
from io import StringIO
import sys
def primaryExpression_sempred(self, localctx: PrimaryExpressionContext, predIndex: int):
    if predIndex == 12:
        return self.precpred(self._ctx, 8)
    if predIndex == 13:
        return self.precpred(self._ctx, 6)