from antlr4 import *
from io import StringIO
import sys
def createTableClauses(self):
    return self.getTypedRuleContext(fugue_sqlParser.CreateTableClausesContext, 0)