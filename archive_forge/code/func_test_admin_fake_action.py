from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_fake_action(self):
    self.assertRaises(exceptions.CommandFailed, self.nova, 'this-does-nova-exist')