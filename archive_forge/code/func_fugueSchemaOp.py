from antlr4 import *
from io import StringIO
import sys
def fugueSchemaOp(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueSchemaOpContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueSchemaOpContext, i)