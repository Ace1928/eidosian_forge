from antlr4 import *
from io import StringIO
import sys
@property
def allUpperCase(self):
    if '_all_upper_case' in self.__dict__:
        return self._all_upper_case
    return False