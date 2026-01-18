from antlr4 import *
from io import StringIO
import sys
def fugueFileFormat(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueFileFormatContext, 0)