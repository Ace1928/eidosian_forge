import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSortConsidersDependencies(self):
    """Tests with different dependencies are sorted together."""
    resource_one = testresources.TestResource()
    resource_two = testresources.TestResource()
    resource_one_common = testresources.TestResource()
    resource_one_common.setUpCost = 2
    resource_one_common.tearDownCost = 2
    resource_two_common = testresources.TestResource()
    resource_two_common.setUpCost = 2
    resource_two_common.tearDownCost = 2
    dep = testresources.TestResource()
    dep.setUpCost = 20
    dep.tearDownCost = 20
    resource_one.resources.append(('dep1', dep))
    resource_two.resources.append(('dep2', dep))
    self.case1.resources = [('withdep', resource_one), ('common', resource_one_common)]
    self.case2.resources = [('withdep', resource_two), ('common', resource_two_common)]
    self.case3.resources = [('_one', resource_one_common), ('_two', resource_two_common)]
    self.case4.resources = []
    acceptable_orders = [[self.case1, self.case2, self.case3, self.case4], [self.case2, self.case1, self.case3, self.case4], [self.case3, self.case1, self.case2, self.case4], [self.case3, self.case2, self.case1, self.case4]]
    for permutation in self._permute_four(self.cases):
        self.assertIn(self.sortTests(permutation), acceptable_orders)