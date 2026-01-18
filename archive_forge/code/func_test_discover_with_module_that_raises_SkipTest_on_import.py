import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def test_discover_with_module_that_raises_SkipTest_on_import(self):
    if not unittest.BaseTestSuite._cleanup:
        raise unittest.SkipTest('Suite cleanup is disabled')
    loader = unittest.TestLoader()

    def _get_module_from_name(name):
        raise unittest.SkipTest('skipperoo')
    loader._get_module_from_name = _get_module_from_name
    self.setup_import_issue_tests('test_skip_dummy.py')
    suite = loader.discover('.')
    self.assertEqual(suite.countTestCases(), 1)
    result = unittest.TestResult()
    suite.run(result)
    self.assertEqual(len(result.skipped), 1)
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        pickle.loads(pickle.dumps(suite, proto))