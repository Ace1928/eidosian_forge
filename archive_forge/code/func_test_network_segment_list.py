import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_segment_list(self):
    name = uuid.uuid4().hex
    json_output = self.openstack(' network segment create ' + '--network ' + self.NETWORK_ID + ' ' + '--network-type geneve ' + '--segment 2055 ' + name, parse_output=True)
    network_segment_id = json_output.get('id')
    network_segment_name = json_output.get('name')
    self.addCleanup(self.openstack, 'network segment delete ' + network_segment_id)
    self.assertEqual(name, json_output['name'])
    json_output = self.openstack('network segment list', parse_output=True)
    item_map = {item.get('ID'): item.get('Name') for item in json_output}
    self.assertIn(network_segment_id, item_map.keys())
    self.assertIn(network_segment_name, item_map.values())