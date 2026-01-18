import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_name_collision(self):
    proc = lazy_import.ImportProcessor()
    proc._build_map('import foo')
    self.assertRaises(lazy_import.ImportNameCollision, proc._build_map, 'import bar as foo')
    self.assertRaises(lazy_import.ImportNameCollision, proc._build_map, 'from foo import bar as foo')
    self.assertRaises(lazy_import.ImportNameCollision, proc._build_map, 'from bar import foo')