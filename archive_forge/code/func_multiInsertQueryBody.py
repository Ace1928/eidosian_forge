from antlr4 import *
from io import StringIO
import sys
def multiInsertQueryBody(self, i: int=None):
    if i is None:
        return self.getTypedRuleContexts(fugue_sqlParser.MultiInsertQueryBodyContext)
    else:
        return self.getTypedRuleContext(fugue_sqlParser.MultiInsertQueryBodyContext, i)