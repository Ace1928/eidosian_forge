from openstackclient.common import availability_zone
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def test_availability_zone_list_network(self):
    arglist = ['--network']
    verifylist = [('network', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.compute_sdk_client.availability_zones.assert_not_called()
    self.volume_sdk_client.availability_zones.assert_not_called()
    self.network_client.availability_zones.assert_called_with()
    self.assertEqual(self.short_columnslist, columns)
    datalist = ()
    for network_az in self.network_azs:
        datalist += _build_network_az_datalist(network_az)
    self.assertEqual(datalist, tuple(data))