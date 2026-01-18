from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_login_with(self):
    """
        L{FakeLaunchpad.login_with} ignores all parameters and returns a new
        instance using the builtin WADL definition.
        """
    launchpad = FakeLaunchpad.login_with('name')
    self.assertTrue(isinstance(launchpad, FakeLaunchpad))