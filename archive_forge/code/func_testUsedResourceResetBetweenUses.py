from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testUsedResourceResetBetweenUses(self):
    resource_manager = MockResettableResource()
    resource_manager.getResource()
    resource = resource_manager.getResource()
    resource_manager.dirtied(resource)
    resource_manager.finishedWith(resource)
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    resource_manager.finishedWith(resource)
    self.assertEqual(1, resource_manager.makes)
    self.assertEqual(1, resource_manager.resets)
    self.assertEqual(1, resource_manager.cleans)