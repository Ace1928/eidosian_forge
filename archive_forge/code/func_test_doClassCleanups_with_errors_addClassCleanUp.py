import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_doClassCleanups_with_errors_addClassCleanUp(self):

    class TestableTest(unittest.TestCase):

        def testNothing(self):
            pass

    def cleanup1():
        raise CustomError('cleanup1')

    def cleanup2():
        raise CustomError('cleanup2')
    TestableTest.addClassCleanup(cleanup1)
    TestableTest.addClassCleanup(cleanup2)
    TestableTest.doClassCleanups()
    self.assertEqual(len(TestableTest.tearDown_exceptions), 2)
    e1, e2 = TestableTest.tearDown_exceptions
    self.assertIsInstance(e1[1], CustomError)
    self.assertEqual(str(e1[1]), 'cleanup2')
    self.assertIsInstance(e2[1], CustomError)
    self.assertEqual(str(e2[1]), 'cleanup1')