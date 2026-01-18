from antlr4 import *
from io import StringIO
import sys
def fileFormat(self):
    return self.getTypedRuleContext(fugue_sqlParser.FileFormatContext, 0)