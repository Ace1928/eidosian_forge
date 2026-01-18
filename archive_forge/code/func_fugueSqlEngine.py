from antlr4 import *
from io import StringIO
import sys
def fugueSqlEngine(self):
    return self.getTypedRuleContext(fugue_sqlParser.FugueSqlEngineContext, 0)