import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testGetSubTestDescriptionWithoutDocstring(self):
    with self.subTest(foo=1, bar=2):
        result = unittest.TextTestResult(None, True, 1)
        self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithoutDocstring) (foo=1, bar=2)')
    with self.subTest('some message'):
        result = unittest.TextTestResult(None, True, 1)
        self.assertEqual(result.getDescription(self._subtest), 'testGetSubTestDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetSubTestDescriptionWithoutDocstring) [some message]')