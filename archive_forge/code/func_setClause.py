from antlr4 import *
from io import StringIO
import sys
def setClause(self):
    return self.getTypedRuleContext(fugue_sqlParser.SetClauseContext, 0)