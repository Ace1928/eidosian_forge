import unittest
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_meter_rule_show(self):
    """Test create, show, delete"""
    json_output = self.openstack('network meter rule create ' + '--remote-ip-prefix 10.0.0.0/8 ' + '--egress ' + self.METER_ID, parse_output=True)
    rule_id = json_output.get('id')
    self.assertEqual('egress', json_output.get('direction'))
    json_output = self.openstack('network meter rule show ' + rule_id, parse_output=True)
    self.assertEqual('10.0.0.0/8', json_output.get('remote_ip_prefix'))
    self.assertIsNotNone(rule_id)
    self.addCleanup(self.openstack, 'network meter rule delete ' + rule_id)