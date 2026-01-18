import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
def testResourcedCaseWithNoResources(self):
    resourced_case = self.makeResourcedTestCase(has_resource=False)
    resource_set_tests = split_by_resources([resourced_case])
    self.assertEqual({frozenset(): [resourced_case]}, resource_set_tests)