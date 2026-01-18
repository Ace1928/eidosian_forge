import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def checkRecursion(self, parseElementList):
    if self in parseElementList:
        raise RecursiveGrammarException(parseElementList + [self])
    subRecCheckList = parseElementList[:] + [self]
    if self.expr is not None:
        self.expr.checkRecursion(subRecCheckList)