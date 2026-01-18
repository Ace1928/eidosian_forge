from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_invalid_property_name(self):
    """
        Sample data set on a L{FakeLaunchpad} instance is validated against
        the WADL definition.  If a key is defined on a resource that doesn't
        match a related parameter, an L{IntegrityError} is raised.
        """
    self.assertRaises(IntegrityError, setattr, self.launchpad, 'me', dict(foo='bar'))