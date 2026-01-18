import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def findTestCases(module, prefix='test', sortUsing=util.three_way_cmp, suiteClass=suite.TestSuite):
    import warnings
    warnings.warn('unittest.findTestCases() is deprecated and will be removed in Python 3.13. Please use unittest.TestLoader.loadTestsFromModule() instead.', DeprecationWarning, stacklevel=2)
    return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromModule(module)