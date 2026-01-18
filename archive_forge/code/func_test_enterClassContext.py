import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_enterClassContext(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass
    cleanups = []
    TestableTest.addClassCleanup(cleanups.append, 'cleanup1')
    cm = TestCM(cleanups, 42)
    self.assertEqual(TestableTest.enterClassContext(cm), 42)
    TestableTest.addClassCleanup(cleanups.append, 'cleanup2')
    TestableTest.doClassCleanups()
    self.assertEqual(cleanups, ['enter', 'cleanup2', 'exit', 'cleanup1'])