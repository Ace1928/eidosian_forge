from antlr4 import *
from io import StringIO
import sys
@property
def isValidDecimal(self):
    return True
    nextChar = self._input.LA(1)
    if nextChar >= 'A' and nextChar <= 'Z' or (nextChar >= '0' and nextChar <= '9') or nextChar == '_':
        return False
    else:
        return True