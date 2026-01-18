from antlr4 import *
from io import StringIO
import sys
def functionIdentifier(self):
    return self.getTypedRuleContext(fugue_sqlParser.FunctionIdentifierContext, 0)