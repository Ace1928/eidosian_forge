import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def testCleanupInRun(self):
    blowUp = False
    ordering = []

    class TestableTest(unittest.TestCase):

        def setUp(self):
            ordering.append('setUp')
            test.addCleanup(cleanup2)
            if blowUp:
                raise CustomError('foo')

        def testNothing(self):
            ordering.append('test')
            test.addCleanup(cleanup3)

        def tearDown(self):
            ordering.append('tearDown')
    test = TestableTest('testNothing')

    def cleanup1():
        ordering.append('cleanup1')

    def cleanup2():
        ordering.append('cleanup2')

    def cleanup3():
        ordering.append('cleanup3')
    test.addCleanup(cleanup1)

    def success(some_test):
        self.assertEqual(some_test, test)
        ordering.append('success')
    result = unittest.TestResult()
    result.addSuccess = success
    test.run(result)
    self.assertEqual(ordering, ['setUp', 'test', 'tearDown', 'cleanup3', 'cleanup2', 'cleanup1', 'success'])
    blowUp = True
    ordering = []
    test = TestableTest('testNothing')
    test.addCleanup(cleanup1)
    test.run(result)
    self.assertEqual(ordering, ['setUp', 'cleanup2', 'cleanup1'])