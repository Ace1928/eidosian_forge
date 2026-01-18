from antlr4 import *
from io import StringIO
import sys
def fugueJsonValue(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.FugueJsonValueContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.FugueJsonValueContext, i)