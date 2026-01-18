from antlr4 import *
from io import StringIO
import sys
def massDecl2(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(AutolevParser.MassDecl2Context)
    else:
        return self.getTypedRuleContext(AutolevParser.MassDecl2Context, i)