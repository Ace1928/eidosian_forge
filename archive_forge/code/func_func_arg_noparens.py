from antlr4 import *
from io import StringIO
import sys
def func_arg_noparens(self):
    return self.getTypedRuleContext(LaTeXParser.Func_arg_noparensContext, 0)