import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testMixThemUp(self):
    normal_cases = [self.makeTestCase() for i in range(3)]
    normal_cases.extend([self.makeResourcedTestCase(has_resource=False) for i in range(3)])
    resourced_cases = [self.makeResourcedTestCase() for i in range(3)]
    all_cases = normal_cases + resourced_cases
    random.shuffle(all_cases)
    resource_set_tests = split_by_resources(all_cases)
    self.assertEqual(set(normal_cases), set(resource_set_tests[frozenset()]))
    for case in resourced_cases:
        resource = case.resources[0][1]
        self.assertEqual([case], resource_set_tests[frozenset([resource])])