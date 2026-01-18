import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def testTestCaseDebugExecutesCleanups(self):
    ordering = []

    class TestableTest(unittest.TestCase):

        def setUp(self):
            ordering.append('setUp')
            self.addCleanup(cleanup1)

        def testNothing(self):
            ordering.append('test')
            self.addCleanup(cleanup3)

        def tearDown(self):
            ordering.append('tearDown')
            test.addCleanup(cleanup4)
    test = TestableTest('testNothing')

    def cleanup1():
        ordering.append('cleanup1')
        test.addCleanup(cleanup2)

    def cleanup2():
        ordering.append('cleanup2')

    def cleanup3():
        ordering.append('cleanup3')

    def cleanup4():
        ordering.append('cleanup4')
    test.debug()
    self.assertEqual(ordering, ['setUp', 'test', 'tearDown', 'cleanup4', 'cleanup3', 'cleanup1', 'cleanup2'])