from antlr4 import *
from io import StringIO
import sys
def relation_sempred(self, localctx: RelationContext, predIndex: int):
    if predIndex == 0:
        return self.precpred(self._ctx, 2)