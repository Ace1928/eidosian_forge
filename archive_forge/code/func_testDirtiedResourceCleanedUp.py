import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testDirtiedResourceCleanedUp(self):
    make_counter = MakeCounter()

    def testOne(test):
        make_counter.calls.append('test one')
        make_counter.dirtied(test._default)

    def testTwo(test):
        make_counter.calls.append('test two')
    case1 = self.makeResourcedTestCase(make_counter, testOne)
    case2 = self.makeResourcedTestCase(make_counter, testTwo)
    self.optimising_suite.addTest(case1)
    self.optimising_suite.addTest(case2)
    result = unittest.TestResult()
    self.optimising_suite.run(result)
    self.assertEqual(result.testsRun, 2)
    self.assertEqual(result.wasSuccessful(), True)
    self.assertEqual(make_counter.calls, [('make', 'boo 1'), 'test one', ('clean', 'boo 1'), ('make', 'boo 2'), 'test two', ('clean', 'boo 2')])