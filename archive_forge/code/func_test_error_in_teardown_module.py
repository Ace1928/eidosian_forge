import io
import sys
import unittest
def test_error_in_teardown_module(self):

    class Module(object):
        moduleTornDown = 0

        @staticmethod
        def tearDownModule():
            Module.moduleTornDown += 1
            raise TypeError('foo')

    class Test(unittest.TestCase):
        classSetUp = False
        classTornDown = False

        @classmethod
        def setUpClass(cls):
            Test.classSetUp = True

        @classmethod
        def tearDownClass(cls):
            Test.classTornDown = True

        def test_one(self):
            pass

        def test_two(self):
            pass

    class Test2(unittest.TestCase):

        def test_one(self):
            pass

        def test_two(self):
            pass
    Test.__module__ = 'Module'
    Test2.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = self.runTests(Test, Test2)
    self.assertEqual(Module.moduleTornDown, 1)
    self.assertEqual(result.testsRun, 4)
    self.assertTrue(Test.classSetUp)
    self.assertTrue(Test.classTornDown)
    self.assertEqual(len(result.errors), 1)
    error, _ = result.errors[0]
    self.assertEqual(str(error), 'tearDownModule (Module)')