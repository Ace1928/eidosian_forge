from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_callable_object_return_None(self):
    """
        A fake method passes through a return value of None rather than
        trying to create a L{FakeResource}.
        """
    self.launchpad.branches = dict(getByUniqueName=lambda name: None)
    self.assertIsNone(self.launchpad.branches.getByUniqueName('foo'))