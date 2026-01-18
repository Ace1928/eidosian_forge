from antlr4 import *
from io import StringIO
import sys
def sortItem(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.SortItemContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.SortItemContext, i)