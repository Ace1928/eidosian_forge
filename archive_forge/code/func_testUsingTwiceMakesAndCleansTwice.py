from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testUsingTwiceMakesAndCleansTwice(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    resource = resource_manager.getResource()
    resource_manager.finishedWith(resource)
    self.assertEqual(2, resource_manager.makes)
    self.assertEqual(2, resource_manager.cleans)