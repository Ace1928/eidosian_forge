from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_lp_save(self):
    """
        Sample object have an C{lp_save} method that is a no-op by default.
        """
    self.launchpad.me = dict(name='foo')
    self.assertTrue(self.launchpad.me.lp_save())