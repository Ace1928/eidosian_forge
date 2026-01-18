import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testGetDescriptionWithoutDocstring(self):
    result = unittest.TextTestResult(None, True, 1)
    self.assertEqual(result.getDescription(self), 'testGetDescriptionWithoutDocstring (' + __name__ + '.Test_TextTestResult.testGetDescriptionWithoutDocstring)')