import uuid
from openstackclient.tests.functional.network.v2 import common
def test_local_ip_set_unset_and_show(self):
    """Tests create options, set, and show"""
    name = uuid.uuid4().hex
    newname = name + '_'
    cmd_output = self.openstack('local ip create ' + '--description aaaa ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'local ip delete ' + newname)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual('aaaa', cmd_output['description'])
    raw_output = self.openstack('local ip set ' + '--name ' + newname + ' ' + '--description bbbb ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('local ip show ' + newname, parse_output=True)
    self.assertEqual(newname, cmd_output['name'])
    self.assertEqual('bbbb', cmd_output['description'])