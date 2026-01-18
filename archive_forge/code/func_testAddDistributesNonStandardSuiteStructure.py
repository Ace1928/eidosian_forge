import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testAddDistributesNonStandardSuiteStructure(self):
    case1 = self.makeTestCase()
    case2 = self.makeTestCase()
    inner_suite = unittest.TestSuite([case2])
    suite = CustomSuite([case1, inner_suite])
    self.optimising_suite.addTest(suite)
    self.assertEqual([CustomSuite([case1]), CustomSuite([inner_suite])], self.optimising_suite._tests)