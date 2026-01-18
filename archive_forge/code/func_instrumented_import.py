import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def instrumented_import(mod, scope1, scope2, fromlist, level):
    self.actions.append(('import', mod, fromlist, level))
    return __import__(mod, scope1, scope2, fromlist, level=level)