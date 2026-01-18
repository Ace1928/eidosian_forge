from tempest.lib import decorators
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_admin_invalid_bypass_url(self):
    self.assertRaises(exceptions.CommandFailed, self.nova, 'list', flags='--os-endpoint-override badurl')