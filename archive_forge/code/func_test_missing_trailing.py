import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_missing_trailing(self):
    proc = lazy_import.ImportProcessor()
    self.assertRaises(lazy_import.InvalidImportLine, proc._canonicalize_import_text, 'from foo import (\n  bar\n')