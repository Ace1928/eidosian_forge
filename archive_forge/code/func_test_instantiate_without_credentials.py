from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_instantiate_without_credentials(self):
    """
        A L{FakeLaunchpad} instantiated without credentials has its
        C{credentials} attribute set to C{None}.
        """
    self.assertIsNone(self.launchpad.credentials)