import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def checkUnindent(s, l, t):
    if l >= len(s):
        return
    curCol = col(l, s)
    if not (indentStack and curCol < indentStack[-1] and (curCol <= indentStack[-2])):
        raise ParseException(s, l, 'not an unindent')
    indentStack.pop()