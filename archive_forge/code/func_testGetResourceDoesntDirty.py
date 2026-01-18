from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testGetResourceDoesntDirty(self):
    resource_manager = MockResource()
    resource_manager.getResource()
    self.assertEqual(resource_manager._dirty, False)