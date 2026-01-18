from antlr4 import *
from io import StringIO
import sys
def orderedIdentifier(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.OrderedIdentifierContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.OrderedIdentifierContext, i)