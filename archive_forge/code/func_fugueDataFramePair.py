from antlr4 import *
from io import StringIO
import sys
def fugueDataFramePair(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueDataFramePairContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueDataFramePairContext, i)