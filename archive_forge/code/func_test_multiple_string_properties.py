from datetime import datetime
from testresources import ResourcedTestCase
from launchpadlib.testing.launchpad import (
from launchpadlib.testing.resources import (
def test_multiple_string_properties(self):
    """
        Sample data can be created by setting L{FakeLaunchpad} attributes with
        dicts that represent objects.
        """
    self.launchpad.me = dict(name='foo', display_name='Foo')
    self.assertEqual('foo', self.launchpad.me.name)
    self.assertEqual('Foo', self.launchpad.me.display_name)