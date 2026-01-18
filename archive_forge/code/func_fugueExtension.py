from antlr4 import *
from io import StringIO
import sys
def fugueExtension(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueExtensionContext, 0)