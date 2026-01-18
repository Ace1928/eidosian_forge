import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSingleCaseResourceAcquisition(self):
    sample_resource = MakeCounter()

    def getResourceCount(test):
        self.assertEqual(sample_resource._uses, 2)
    case = self.makeResourcedTestCase(sample_resource, getResourceCount)
    self.optimising_suite.addTest(case)
    result = unittest.TestResult()
    self.optimising_suite.run(result)
    self.assertEqual(result.testsRun, 1)
    self.assertEqual(result.wasSuccessful(), True)
    self.assertEqual(sample_resource._uses, 0)