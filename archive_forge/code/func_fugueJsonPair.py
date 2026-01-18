from antlr4 import *
from io import StringIO
import sys
def fugueJsonPair(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueJsonPairContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonPairContext, i)