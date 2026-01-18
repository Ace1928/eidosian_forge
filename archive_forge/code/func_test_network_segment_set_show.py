import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_segment_set_show(self):
    name = uuid.uuid4().hex
    json_output = self.openstack(' network segment create ' + '--network ' + self.NETWORK_ID + ' ' + '--network-type geneve ' + '--segment 2055 ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'network segment delete ' + name)
    if self.is_extension_enabled('standard-attr-segment'):
        self.assertEqual('', json_output['description'])
    else:
        self.assertIsNone(json_output['description'])
    new_description = 'new_description'
    cmd_output = self.openstack('network segment set ' + '--description ' + new_description + ' ' + name)
    self.assertOutput('', cmd_output)
    json_output = self.openstack('network segment show ' + name, parse_output=True)
    self.assertEqual(new_description, json_output['description'])