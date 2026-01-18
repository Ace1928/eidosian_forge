from antlr4 import *
from io import StringIO
import sys
def namedWindow(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.NamedWindowContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.NamedWindowContext, i)