import unittest2 as unittest
from mock import sentinel, DEFAULT
def testDEFAULT(self):
    self.assertIs(DEFAULT, sentinel.DEFAULT)