from antlr4 import *
from io import StringIO
import sys
def constantList(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.ConstantListContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.ConstantListContext, i)