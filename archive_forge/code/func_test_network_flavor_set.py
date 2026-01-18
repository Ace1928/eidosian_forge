import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_flavor_set(self):
    """Tests create options, set, show, delete"""
    name = uuid.uuid4().hex
    newname = name + '_'
    cmd_output = self.openstack('network flavor create --description testdescription --disable --service-type  L3_ROUTER_NAT ' + name, parse_output=True)
    self.addCleanup(self.openstack, 'network flavor delete ' + newname)
    self.assertEqual(name, cmd_output['name'])
    self.assertEqual(False, cmd_output['enabled'])
    self.assertEqual('testdescription', cmd_output['description'])
    raw_output = self.openstack('network flavor set --name ' + newname + ' --disable ' + name)
    self.assertOutput('', raw_output)
    cmd_output = self.openstack('network flavor show ' + newname, parse_output=True)
    self.assertEqual(newname, cmd_output['name'])
    self.assertEqual(False, cmd_output['enabled'])
    self.assertEqual('testdescription', cmd_output['description'])