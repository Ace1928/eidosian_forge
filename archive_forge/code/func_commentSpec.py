from antlr4 import *
from io import StringIO
import sys
def commentSpec(self):
    return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, 0)