from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_invalid_callable_object_return_type(self):
    """
        An L{IntegrityError} is raised if a method returns an invalid result.
        """
    branches = dict(total_size=8)
    self.launchpad.me = dict(getBranches=lambda statuses: branches)
    self.assertRaises(IntegrityError, self.launchpad.me.getBranches, [])