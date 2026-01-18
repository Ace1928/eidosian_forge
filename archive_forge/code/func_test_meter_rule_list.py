import unittest
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_meter_rule_list(self):
    """Test create, list, delete"""
    json_output = self.openstack('network meter rule create ' + '--remote-ip-prefix 10.0.0.0/8 ' + self.METER_ID, parse_output=True)
    rule_id_1 = json_output.get('id')
    self.addCleanup(self.openstack, 'network meter rule delete ' + rule_id_1)
    self.assertEqual('10.0.0.0/8', json_output.get('remote_ip_prefix'))
    json_output_1 = self.openstack('network meter rule create ' + '--remote-ip-prefix 11.0.0.0/8 ' + self.METER_ID, parse_output=True)
    rule_id_2 = json_output_1.get('id')
    self.addCleanup(self.openstack, 'network meter rule delete ' + rule_id_2)
    self.assertEqual('11.0.0.0/8', json_output_1.get('remote_ip_prefix'))
    json_output = self.openstack('network meter rule list', parse_output=True)
    rule_id_list = [item.get('ID') for item in json_output]
    ip_prefix_list = [item.get('Remote IP Prefix') for item in json_output]
    self.assertIn(rule_id_1, rule_id_list)
    self.assertIn(rule_id_2, rule_id_list)
    self.assertIn('10.0.0.0/8', ip_prefix_list)
    self.assertIn('11.0.0.0/8', ip_prefix_list)