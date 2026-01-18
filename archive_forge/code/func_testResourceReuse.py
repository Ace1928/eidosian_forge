import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testResourceReuse(self):
    make_counter = MakeCounter()

    def getResourceCount(test):
        self.assertEqual(make_counter._uses, 2)
    case = self.makeResourcedTestCase(make_counter, getResourceCount)
    case2 = self.makeResourcedTestCase(make_counter, getResourceCount)
    self.optimising_suite.addTest(case)
    self.optimising_suite.addTest(case2)
    result = unittest.TestResult()
    self.optimising_suite.run(result)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual(result.wasSuccessful(), True)
    self.assertEqual(make_counter._uses, 0)
    self.assertEqual(make_counter.makes, 1)
    self.assertEqual(make_counter.cleans, 1)