import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testAddTestSuite(self):
    case = self.makeTestCase()
    suite = unittest.TestSuite([case])
    self.optimising_suite.addTest(suite)
    self.assertEqual([case], self.optimising_suite._tests)