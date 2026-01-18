import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testAddFlattensStandardSuiteStructure(self):
    case1 = self.makeTestCase()
    case2 = self.makeTestCase()
    case3 = self.makeTestCase()
    suite = unittest.TestSuite([unittest.TestSuite([case1, unittest.TestSuite([case2])]), case3])
    self.optimising_suite.addTest(suite)
    self.assertEqual([case1, case2, case3], self.optimising_suite._tests)