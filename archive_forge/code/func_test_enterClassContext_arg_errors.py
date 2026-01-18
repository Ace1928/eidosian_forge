import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_enterClassContext_arg_errors(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        TestableTest.enterClassContext(LacksEnterAndExit())
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        TestableTest.enterClassContext(LacksEnter())
    with self.assertRaisesRegex(TypeError, 'the context manager'):
        TestableTest.enterClassContext(LacksExit())
    self.assertEqual(TestableTest._class_cleanups, [])