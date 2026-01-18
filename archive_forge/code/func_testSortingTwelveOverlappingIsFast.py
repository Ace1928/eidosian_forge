import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testSortingTwelveOverlappingIsFast(self):
    managers = []
    for pos in range(12):
        managers.append(testresources.TestResourceManager())
    cases = [self.case1, self.case2, self.case3, self.case4]
    for pos in range(5, 13):
        cases.append(testtools.clone_test_with_new_id(cases[0], 'case%d' % pos))
    tempdir = testresources.TestResourceManager()
    for case, manager in zip(cases, managers):
        case.resources = [('_resource', manager), ('tempdir', tempdir)]
    result = self.sortTests(cases)
    self.assertEqual(12, len(result))