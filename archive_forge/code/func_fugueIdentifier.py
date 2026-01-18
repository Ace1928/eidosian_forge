from antlr4 import *
from io import StringIO
import sys
def fugueIdentifier(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueIdentifierContext, 0)