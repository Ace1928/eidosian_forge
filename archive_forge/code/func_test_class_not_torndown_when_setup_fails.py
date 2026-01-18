import io
import sys
import unittest
def test_class_not_torndown_when_setup_fails(self):

    class Test(unittest.TestCase):
        tornDown = False

        @classmethod
        def setUpClass(cls):
            raise TypeError

        @classmethod
        def tearDownClass(cls):
            Test.tornDown = True
            raise TypeError('foo')

        def test_one(self):
            pass
    self.runTests(Test)
    self.assertFalse(Test.tornDown)