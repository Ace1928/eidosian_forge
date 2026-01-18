import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_module_cleanUp_with_multiple_classes(self):
    ordering = []

    def cleanup1():
        ordering.append('cleanup1')

    def cleanup2():
        ordering.append('cleanup2')

    def cleanup3():
        ordering.append('cleanup3')

    class Module(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')
            unittest.addModuleCleanup(cleanup1)

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class TestableTest(unittest.TestCase):

        def setUp(self):
            ordering.append('setUp')
            self.addCleanup(cleanup2)

        def testNothing(self):
            ordering.append('test')

        def tearDown(self):
            ordering.append('tearDown')

    class OtherTestableTest(unittest.TestCase):

        def setUp(self):
            ordering.append('setUp2')
            self.addCleanup(cleanup3)

        def testNothing(self):
            ordering.append('test2')

        def tearDown(self):
            ordering.append('tearDown2')
    TestableTest.__module__ = 'Module'
    OtherTestableTest.__module__ = 'Module'
    sys.modules['Module'] = Module
    runTests(TestableTest, OtherTestableTest)
    self.assertEqual(ordering, ['setUpModule', 'setUp', 'test', 'tearDown', 'cleanup2', 'setUp2', 'test2', 'tearDown2', 'cleanup3', 'tearDownModule', 'cleanup1'])