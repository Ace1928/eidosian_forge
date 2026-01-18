from antlr4 import *
from io import StringIO
import sys
def fromStatementBody(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FromStatementBodyContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FromStatementBodyContext, i)