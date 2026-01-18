from antlr4 import *
from io import StringIO
import sys
def createFileFormat(self):
    return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, 0)