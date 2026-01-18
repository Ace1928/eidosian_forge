from antlr4 import *
from io import StringIO
import sys
def fuguePath(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FuguePathContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FuguePathContext, i)