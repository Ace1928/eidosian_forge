import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testMultipleResources(self):
    resource1 = testresources.TestResource()
    resource2 = testresources.TestResource()
    resourced_case = self.makeResourcedTestCase(has_resource=False)
    resourced_case.resources = [('resource1', resource1), ('resource2', resource2)]
    resource_set_tests = split_by_resources([resourced_case])
    self.assertEqual({frozenset(): [], frozenset([resource1, resource2]): [resourced_case]}, resource_set_tests)