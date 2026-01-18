from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_get_undefined_resource(self):
    """
        An L{AttributeError} is raised if an attribute is accessed on a
        L{FakeLaunchpad} instance that doesn't exist.
        """
    self.launchpad.me = dict(display_name='Foo')
    self.assertRaises(AttributeError, getattr, self.launchpad.me, 'name')