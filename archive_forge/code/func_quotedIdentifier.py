from antlr4 import *
from io import StringIO
import sys
def quotedIdentifier(self):
    return self.getTypedRuleContext(fugue_sqlParser.QuotedIdentifierContext, 0)