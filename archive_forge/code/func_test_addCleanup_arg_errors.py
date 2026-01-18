import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_addCleanup_arg_errors(self):
    cleanups = []

    def cleanup(*args, **kwargs):
        cleanups.append((args, kwargs))

    class TestableTest(unittest.TestCase):

        def setUp(self2):
            self2.addCleanup(cleanup, 1, 2, function=3, self=4)
            with self.assertRaises(TypeError):
                self2.addCleanup(function=cleanup, arg='hello')

        def testNothing(self):
            pass
    with self.assertRaises(TypeError):
        TestableTest().addCleanup()
    with self.assertRaises(TypeError):
        unittest.TestCase.addCleanup(self=TestableTest(), function=cleanup)
    runTests(TestableTest)
    self.assertEqual(cleanups, [((1, 2), {'function': 3, 'self': 4})])