from antlr4 import *
from io import StringIO
import sys
def createTableHeader(self):
    return self.getTypedRuleContext(fugue_sqlParser.CreateTableHeaderContext, 0)