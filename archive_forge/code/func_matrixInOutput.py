from antlr4 import *
from io import StringIO
import sys
def matrixInOutput(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(AutolevParser.MatrixInOutputContext)
    else:
        return self.getTypedRuleContext(AutolevParser.MatrixInOutputContext, i)