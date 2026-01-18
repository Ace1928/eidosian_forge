from antlr4 import *
from io import StringIO
import sys
def mathit_text(self):
    return self.getTypedRuleContext(LaTeXParser.Mathit_textContext, 0)