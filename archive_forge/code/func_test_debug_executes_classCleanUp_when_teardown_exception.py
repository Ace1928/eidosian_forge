import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_debug_executes_classCleanUp_when_teardown_exception(self):
    ordering = []
    blowUp = False

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering, blowUp=blowUp)

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
            raise CustomError('TearDownClassExc')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
    with self.assertRaises(CustomError) as cm:
        suite.debug()
    self.assertEqual(str(cm.exception), 'TearDownClassExc')
    self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass'])
    self.assertTrue(TestableTest._class_cleanups)
    TestableTest._class_cleanups.clear()
    ordering = []
    blowUp = True
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestableTest)
    with self.assertRaises(CustomError) as cm:
        suite.debug()
    self.assertEqual(str(cm.exception), 'TearDownClassExc')
    self.assertEqual(ordering, ['setUpClass', 'test', 'tearDownClass'])
    self.assertTrue(TestableTest._class_cleanups)
    TestableTest._class_cleanups.clear()