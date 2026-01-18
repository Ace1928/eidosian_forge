import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSortIsStableWithinGroups(self):
    """Tests with the same resources maintain their relative order."""
    resource_one = testresources.TestResource()
    resource_two = testresources.TestResource()
    self.case1.resources = [('_one', resource_one)]
    self.case2.resources = [('_one', resource_one)]
    self.case3.resources = [('_one', resource_one), ('_two', resource_two)]
    self.case4.resources = [('_one', resource_one), ('_two', resource_two)]
    for permutation in self._permute_four(self.cases):
        sorted = self.sortTests(permutation)
        self.assertEqual(permutation.index(self.case1) < permutation.index(self.case2), sorted.index(self.case1) < sorted.index(self.case2))
        self.assertEqual(permutation.index(self.case3) < permutation.index(self.case4), sorted.index(self.case3) < sorted.index(self.case4))