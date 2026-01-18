from antlr4 import *
from io import StringIO
import sys
def fugueParams(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueParamsContext, 0)