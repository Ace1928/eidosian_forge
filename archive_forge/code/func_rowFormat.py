from antlr4 import *
from io import StringIO
import sys
def rowFormat(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)