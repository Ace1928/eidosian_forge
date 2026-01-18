import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSortTestsCalled(self):

    class MockOptimisingTestSuite(testresources.OptimisingTestSuite):

        def sortTests(self):
            self.sorted = True
    suite = MockOptimisingTestSuite()
    suite.sorted = False
    suite.run(None)
    self.assertEqual(suite.sorted, True)