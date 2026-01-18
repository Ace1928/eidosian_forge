from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testInitiallyNoCurrentResource(self):
    resource_manager = testresources.TestResource()
    self.assertEqual(None, resource_manager._currentResource)