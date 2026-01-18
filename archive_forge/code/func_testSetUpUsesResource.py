import unittest
import testtools
import testresources
from testresources.tests import ResultWithResourceExtensions
def testSetUpUsesResource(self):
    self.resourced_case.resources = [('foo', self.resource_manager)]
    testresources.setUpResources(self.resourced_case, self.resourced_case.resources, None)
    self.assertEqual(self.resource_manager._uses, 1)