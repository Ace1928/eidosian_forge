import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_incorrect_line(self):
    proc = lazy_import.ImportProcessor()
    self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'foo bar baz')
    self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'improt foo')
    self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'importfoo')
    self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'fromimport')