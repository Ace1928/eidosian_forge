from antlr4 import *
from io import StringIO
import sys
def eval_at(self):
    return self.getTypedRuleContext(LaTeXParser.Eval_atContext, 0)