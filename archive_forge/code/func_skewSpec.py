from antlr4 import *
from io import StringIO
import sys
def skewSpec(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.SkewSpecContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.SkewSpecContext, i)