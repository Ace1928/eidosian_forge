import unittest
import gc
import sys
import weakref
from unittest.test.support import LoggingResult, TestEquality
def test_basetestsuite(self):

    class Test(unittest.TestCase):
        wasSetUp = False
        wasTornDown = False

        @classmethod
        def setUpClass(cls):
            cls.wasSetUp = True

        @classmethod
        def tearDownClass(cls):
            cls.wasTornDown = True

        def testPass(self):
            pass

        def testFail(self):
            fail

    class Module(object):
        wasSetUp = False
        wasTornDown = False

        @staticmethod
        def setUpModule():
            Module.wasSetUp = True

        @staticmethod
        def tearDownModule():
            Module.wasTornDown = True
    Test.__module__ = 'Module'
    sys.modules['Module'] = Module
    self.addCleanup(sys.modules.pop, 'Module')
    suite = unittest.BaseTestSuite()
    suite.addTests([Test('testPass'), Test('testFail')])
    self.assertEqual(suite.countTestCases(), 2)
    result = unittest.TestResult()
    suite.run(result)
    self.assertFalse(Module.wasSetUp)
    self.assertFalse(Module.wasTornDown)
    self.assertFalse(Test.wasSetUp)
    self.assertFalse(Test.wasTornDown)
    self.assertEqual(len(result.errors), 1)
    self.assertEqual(len(result.failures), 0)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual(suite.countTestCases(), 2)