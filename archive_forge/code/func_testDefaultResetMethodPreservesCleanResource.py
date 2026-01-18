from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testDefaultResetMethodPreservesCleanResource(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    self.assertEqual(1, resource_manager.makes)
    self.assertEqual(False, resource_manager._dirty)
    resource_manager.reset(resource)
    self.assertEqual(1, resource_manager.makes)
    self.assertEqual(0, resource_manager.cleans)