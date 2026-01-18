import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_from_one_import_two_two(self):
    exp = {'two': (['one'], 'two', {})}
    self.check(exp, 'from one import two\n')
    self.check(exp, 'from one import (two)\n')
    self.check(exp, 'from one import (two,)\n')
    self.check(exp, 'from one import two as two\n')
    self.check(exp, 'from one import (\n    two,\n    )\n')