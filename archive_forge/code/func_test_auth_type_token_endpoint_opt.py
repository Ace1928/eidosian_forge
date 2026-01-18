from tempest.lib import exceptions as tempest_exc
from openstackclient.tests.functional import base
def test_auth_type_token_endpoint_opt(self):
    try:
        self.openstack('configuration show --os-auth-type token_endpoint', cloud=None)
    except tempest_exc.CommandFailed as e:
        self.assertIn('--os-auth-type: invalid choice:', str(e))
        self.assertIn('token_endpoint', str(e))
    else:
        self.fail('CommandFailed should be raised')