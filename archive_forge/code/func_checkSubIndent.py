import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def checkSubIndent(s, l, t):
    curCol = col(l, s)
    if curCol > indentStack[-1]:
        indentStack.append(curCol)
    else:
        raise ParseException(s, l, 'not a subentry')