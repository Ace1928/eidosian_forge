import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def setDebug(self, flag=True):
    """Enable display of debugging messages while doing pattern matching.
           Set C{flag} to True to enable, False to disable."""
    if flag:
        self.setDebugActions(_defaultStartDebugAction, _defaultSuccessDebugAction, _defaultExceptionDebugAction)
    else:
        self.debug = False
    return self