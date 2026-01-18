from antlr4 import *
from io import StringIO
import sys
def func_arg(self):
    return self.getTypedRuleContext(LaTeXParser.Func_argContext, 0)