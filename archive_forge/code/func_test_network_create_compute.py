import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_create_compute(self):
    """Test Nova-net create options, delete"""
    if self.haz_network:
        self.skipTest('Skip Nova-net test')
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('network create ' + '--subnet 1.2.3.4/28 ' + name1, parse_output=True)
    self.addCleanup(self.openstack, 'network delete ' + name1)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name1, cmd_output['label'])
    self.assertEqual('1.2.3.0/28', cmd_output['cidr'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('network create ' + '--subnet 1.2.4.4/28 ' + '--share ' + name2, parse_output=True)
    self.addCleanup(self.openstack, 'network delete ' + name2)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(name2, cmd_output['label'])
    self.assertEqual('1.2.4.0/28', cmd_output['cidr'])
    self.assertTrue(cmd_output['share_address'])