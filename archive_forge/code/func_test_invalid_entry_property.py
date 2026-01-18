from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_invalid_entry_property(self):
    """
        Sample data for linked entries is validated.
        """
    bug = dict(owner=dict(foo='bar'))
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'bugs', dict(entries=[bug]))