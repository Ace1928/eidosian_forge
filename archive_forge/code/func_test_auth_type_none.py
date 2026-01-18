from tempest.lib import exceptions as tempest_exc
from openstackclient.tests.functional import base
def test_auth_type_none(self):
    cmd_output = self.openstack('configuration show', cloud=None, parse_output=True)
    self.assertIsNotNone(cmd_output)
    self.assertIn('auth_type', cmd_output.keys())
    self.assertEqual('none', cmd_output['auth_type'])