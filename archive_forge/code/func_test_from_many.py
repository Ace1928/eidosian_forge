import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_from_many(self):
    exp = {'two': (['one'], 'two', {}), 'three': (['one', 'two'], 'three', {}), 'five': (['one', 'two'], 'four', {})}
    self.check(exp, 'from one import two\nfrom one.two import three, four as five\n')
    self.check(exp, 'from one import two\nfrom one.two import (\n    three,\n    four as five,\n    )\n')