import uuid
from openstackclient.tests.functional.network.v2 import common
def test_address_group_set_unset_and_show(self):
    """Tests create options, set, unset, and show"""
    name = uuid.uuid4().hex
    newname = name + '_'
    cmd_output = self.openstack('address group create ' + '--description aaaa ' + '--address 10.0.0.1 --address 2001::/16 ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'address group delete ' + newname)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual('aaaa', cmd_output['description'])
    self.assertEqual(2, len(cmd_output['addresses']))
    raw_output = self.openstack('address group set ' + '--name ' + newname + ' ' + '--description bbbb ' + '--address 10.0.0.2 --address 192.0.0.0/8 ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('address group show ' + newname, parse_output=True)
    self.assertEqual(newname, cmd_output['name'])
    self.assertEqual('bbbb', cmd_output['description'])
    self.assertEqual(4, len(cmd_output['addresses']))
    raw_output = self.openstack('address group unset ' + '--address 10.0.0.1 --address 2001::/16 ' + '--address 10.0.0.2 --address 192.0.0.0/8 ' + newname)
    self.assertEqual('', raw_output)
    cmd_output = self.openstack('address group show ' + newname, parse_output=True)
    self.assertEqual(0, len(cmd_output['addresses']))