import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_rbac_show(self):
    cmd_output = self.openstack('network rbac show ' + self.ID, parse_output=True)
    self.assertEqual(self.ID, cmd_output['id'])