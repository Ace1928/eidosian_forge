from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_callable_object_return_type(self):
    """
        The result of a fake method is a L{FakeResource}, automatically
        created from the object used to define the return object.
        """
    branches = dict(total_size='8')
    self.launchpad.me = dict(getBranches=lambda statuses: branches)
    branches = self.launchpad.me.getBranches([])
    self.assertTrue(isinstance(branches, FakeResource))
    self.assertEqual('8', branches.total_size)