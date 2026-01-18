from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetResourceReturnsMakeResource(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    self.assertEqual(resource_manager.make({}), resource)