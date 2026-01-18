import uuid
from openstackclient.tests.functional.network.v2 import common
def test_address_scope_set(self):
    """Tests create options, set, show, delete"""
    name = uuid.uuid4().hex
    newname = name + '_'
    cmd_output = self.openstack('address scope create ' + '--ip-version 4 ' + '--no-share ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'address scope delete ' + newname)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual(4, cmd_output['ip_version'])
    self.assertFalse(cmd_output['shared'])
    raw_output = self.openstack('address scope set ' + '--name ' + newname + ' --share ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('address scope show ' + newname, parse_output=True)
    self.assertEqual(newname, cmd_output['name'])
    self.assertEqual(4, cmd_output['ip_version'])
    self.assertTrue(cmd_output['shared'])