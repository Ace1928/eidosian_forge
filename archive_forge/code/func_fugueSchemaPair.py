from antlr4 import *
from io import StringIO
import sys
def fugueSchemaPair(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaPairContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaPairContext, i)