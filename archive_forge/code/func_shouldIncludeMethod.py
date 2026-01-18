import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def shouldIncludeMethod(attrname):
    if not attrname.startswith(self.testMethodPrefix):
        return False
    testFunc = getattr(testCaseClass, attrname)
    if not callable(testFunc):
        return False
    fullName = f'%s.%s.%s' % (testCaseClass.__module__, testCaseClass.__qualname__, attrname)
    return self.testNamePatterns is None or any((fnmatchcase(fullName, pattern) for pattern in self.testNamePatterns))