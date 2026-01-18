from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testInitiallyUnused(self):
    resource_manager = testresources.TestResource()
    self.assertEqual(0, resource_manager._uses)