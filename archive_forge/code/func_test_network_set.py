import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_set(self):
    """Tests create options, set, show, delete"""
    if not self.haz_network:
        self.skipTest('No Network service present')
    name = uuid.uuid4().hex
    cmd_output = self.openstack('network create --description aaaa --enable --no-share --internal --no-default --enable-port-security %s' % name, parse_output=True)
    self.addCleanup(self.openstack, 'network delete %s' % name)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual('aaaa', cmd_output['description'])
    self.assertEqual(True, cmd_output['admin_state_up'])
    self.assertFalse(cmd_output['shared'])
    self.assertEqual(False, cmd_output['router:external'])
    self.assertFalse(cmd_output['is_default'])
    self.assertTrue(cmd_output['port_security_enabled'])
    raw_output = self.openstack('network set --description cccc --disable --share --external --disable-port-security %s' % name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('network show ' + name, parse_output=True)
    self.assertEqual('cccc', cmd_output['description'])
    self.assertEqual(False, cmd_output['admin_state_up'])
    self.assertTrue(cmd_output['shared'])
    self.assertEqual(True, cmd_output['router:external'])
    self.assertFalse(cmd_output['is_default'])
    self.assertFalse(cmd_output['port_security_enabled'])