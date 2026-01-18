import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_addClassCleanup_arg_errors(self):
    cleanups = []

    def cleanup(*args, **kwargs):
        cleanups.append((args, kwargs))

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.addClassCleanup(cleanup, 1, 2, function=3, cls=4)
            with self.assertRaises(TypeError):
                cls.addClassCleanup(function=cleanup, arg='hello')

        def testNothing(self):
            pass
    with self.assertRaises(TypeError):
        TestableTest.addClassCleanup()
    with self.assertRaises(TypeError):
        unittest.TestCase.addCleanup(cls=TestableTest(), function=cleanup)
    runTests(TestableTest)
    self.assertEqual(cleanups, [((1, 2), {'function': 3, 'cls': 4})])