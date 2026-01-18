from antlr4 import *
from io import StringIO
import sys
def booleanExpression_sempred(self, localctx: BooleanExpressionContext, predIndex: int):
    if predIndex == 4:
        return self.precpred(self._ctx, 2)
    if predIndex == 5:
        return self.precpred(self._ctx, 1)