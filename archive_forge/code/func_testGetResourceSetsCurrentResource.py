from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetResourceSetsCurrentResource(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    self.assertIs(resource_manager._currentResource, resource)