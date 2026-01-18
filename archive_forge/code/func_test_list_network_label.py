from unittest import mock
from openstack.compute.v2 import server_ip
from openstack.tests.unit import base
def test_list_network_label(self):
    label = 'label1'
    sess = mock.Mock()
    resp = mock.Mock()
    sess.get.return_value = resp
    resp.json.return_value = {label: [{'version': 1, 'addr': 'a1'}, {'version': 2, 'addr': 'a2'}]}
    ips = list(server_ip.ServerIP.list(sess, server_id=IDENTIFIER, network_label=label))
    self.assertEqual(2, len(ips))
    ips = sorted(ips, key=lambda ip: ip.version)
    self.assertIsInstance(ips[0], server_ip.ServerIP)
    self.assertEqual(ips[0].network_label, label)
    self.assertEqual(ips[0].address, 'a1')
    self.assertEqual(ips[0].version, 1)
    self.assertIsInstance(ips[1], server_ip.ServerIP)
    self.assertEqual(ips[1].network_label, label)
    self.assertEqual(ips[1].address, 'a2')
    self.assertEqual(ips[1].version, 2)