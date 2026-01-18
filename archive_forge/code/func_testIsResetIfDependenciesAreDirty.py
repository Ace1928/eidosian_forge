from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
def testIsResetIfDependenciesAreDirty(self):
    resource_manager = MockResource()
    dep1 = MockResettableResource()
    resource_manager.resources.append(('dep1', dep1))
    r = resource_manager.getResource()
    dep1.dirtied(r.dep1)
    r = resource_manager.getResource()
    self.assertFalse(resource_manager.isDirty())
    self.assertFalse(dep1.isDirty())
    resource_manager.finishedWith(r)
    resource_manager.finishedWith(r)