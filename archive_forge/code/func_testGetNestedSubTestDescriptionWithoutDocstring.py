import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testGetNestedSubTestDescriptionWithoutDocstring(self):
    with self.subTest(foo=1):
        with self.subTest(baz=2, bar=3):
            result = unittest.TextTestResult(None, True, 1)
            self.assertEqual(result.getDescription(self._subtest), 'testGetNestedSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetNestedSubTestDescriptionWithoutDocstring) (baz=2, bar=3, foo=1)')