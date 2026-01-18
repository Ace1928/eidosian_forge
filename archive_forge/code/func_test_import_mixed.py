import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_mixed(self):
    mixed = {'x': (['one', 'two'], None, {}), 'one': (['one'], None, {'two': (['one', 'two'], None, {})})}
    self.check(mixed, ['import one.two as x, one.two'])
    self.check(mixed, ['import one.two as x', 'import one.two'])
    self.check(mixed, ['import one.two', 'import one.two as x'])