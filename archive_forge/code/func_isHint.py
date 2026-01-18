from antlr4 import *
from io import StringIO
import sys
def isHint(self):
    return False
    nextChar = self._input.LA(1)
    if nextChar == '+':
        return True
    else:
        return False