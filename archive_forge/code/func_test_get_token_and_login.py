from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_get_token_and_login(self):
    """
        L{FakeLaunchpad.get_token_and_login} ignores all parameters and
        returns a new instance using the builtin WADL definition.
        """
    launchpad = FakeLaunchpad.get_token_and_login('name')
    self.assertTrue(isinstance(launchpad, FakeLaunchpad))