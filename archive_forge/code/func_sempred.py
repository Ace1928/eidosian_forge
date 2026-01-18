from antlr4 import *
from io import StringIO
import sys
def sempred(self, localctx: RuleContext, ruleIndex: int, predIndex: int):
    if self._predicates == None:
        self._predicates = dict()
    self._predicates[27] = self.expr_sempred
    pred = self._predicates.get(ruleIndex, None)
    if pred is None:
        raise Exception('No predicate with index:' + str(ruleIndex))
    else:
        return pred(localctx, predIndex)