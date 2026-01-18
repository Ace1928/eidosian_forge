from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testDirtyingResourceTriggersCleanOnGet(self):
    resource_manager = MockResource()
    resource1 = resource_manager.getResource()
    resource2 = resource_manager.getResource()
    resource_manager.dirtied(resource2)
    resource_manager.finishedWith(resource2)
    self.assertEqual(0, resource_manager.cleans)
    resource3 = resource_manager.getResource()
    self.assertEqual(1, resource_manager.cleans)
    resource_manager.finishedWith(resource3)
    resource_manager.finishedWith(resource1)
    self.assertEqual(2, resource_manager.cleans)