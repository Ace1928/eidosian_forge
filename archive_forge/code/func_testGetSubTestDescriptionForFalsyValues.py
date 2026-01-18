import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testGetSubTestDescriptionForFalsyValues(self):
    expected = 'testGetSubTestDescriptionForFalsyValues (%s.Test_TextTestResult.testGetSubTestDescriptionForFalsyValues) [%s]'
    result = unittest.TextTestResult(None, True, 1)
    for arg in [0, None, []]:
        with self.subTest(arg):
            self.assertEqual(result.getDescription(self._subtest), expected % (__name__, arg))