import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_enterContext_arg_errors(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass
    test = TestableTest('testNothing')
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        test.enterContext(LacksEnterAndExit())
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        test.enterContext(LacksEnter())
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        test.enterContext(LacksExit())
    self.assertEqual(test._cleanups, [])