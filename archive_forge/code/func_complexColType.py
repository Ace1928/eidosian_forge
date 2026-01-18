from antlr4 import *
from io import StringIO
import sys
def complexColType(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.ComplexColTypeContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.ComplexColTypeContext, i)