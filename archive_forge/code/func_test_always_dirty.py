from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def test_always_dirty(self):
    fixture = LoggingFixture()
    mgr = testresources.FixtureResource(fixture)
    resource = mgr.getResource()
    self.assertTrue(mgr.isDirty())
    mgr.finishedWith(resource)