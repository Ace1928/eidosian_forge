import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_delete_compute(self):
    """Test create, delete multiple"""
    if self.haz_network:
        self.skipTest('Skip Nova-net test')
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('network create ' + '--subnet 9.8.7.6/28 ' + name1, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name1, cmd_output['label'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('network create ' + '--subnet 8.7.6.5/28 ' + name2, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name2, cmd_output['label'])