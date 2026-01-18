import json
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_trunk_create_delete(self):
    trunk_name = uuid.uuid4().hex
    self.openstack('network trunk create %s --parent-port %s -f json ' % (trunk_name, self.parent_port_name))
    raw_output = self.openstack('network trunk delete ' + trunk_name)
    self.assertEqual('', raw_output)