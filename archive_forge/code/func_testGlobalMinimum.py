import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testGlobalMinimum(self):
    resource_one = testresources.TestResource()
    resource_one.setUpCost = 20
    resource_two = testresources.TestResource()
    resource_two.tearDownCost = 50
    resource_three = testresources.TestResource()
    resource_three.setUpCost = 72
    acceptable_orders = [[self.case1, self.case2, self.case3, self.case4], [self.case1, self.case3, self.case2, self.case4], [self.case2, self.case3, self.case1, self.case4], [self.case3, self.case2, self.case1, self.case4]]
    self.case1.resources = [('_one', resource_one)]
    self.case2.resources = [('_two', resource_two)]
    self.case3.resources = [('_two', resource_two), ('_three', resource_three)]
    for permutation in self._permute_four(self.cases):
        self.assertIn(self.sortTests(permutation), acceptable_orders)