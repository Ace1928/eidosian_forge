from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testFinishedWithMarksNonDirty(self):
    resource_manager = MockResource()
    resource = resource_manager.getResource()
    resource_manager.dirtied(resource)
    resource_manager.finishedWith(resource)
    self.assertEqual(False, resource_manager._dirty)