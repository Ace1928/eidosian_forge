import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def check_result(self, expected, from_strings):
    proc = lazy_import.ImportProcessor()
    for from_str in from_strings:
        proc._convert_from_str(from_str)
    self.assertEqual(expected, proc.imports, 'Import of %r was not converted correctly %s != %s' % (from_strings, expected, proc.imports))