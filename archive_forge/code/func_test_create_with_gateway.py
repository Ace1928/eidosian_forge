from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_with_gateway(self):
    _network = network_fakes.create_one_network()
    _subnet = network_fakes.FakeSubnet.create_one_subnet()
    self.network_client.find_network = mock.Mock(return_value=_network)
    self.network_client.find_subnet = mock.Mock(return_value=_subnet)
    arglist = [self.new_router.name, '--external-gateway', _network.name, '--enable-snat', '--fixed-ip', 'ip-address=2001:db8::1']
    verifylist = [('name', self.new_router.name), ('enable', True), ('distributed', False), ('ha', False), ('external_gateway', _network.name), ('enable_snat', True), ('fixed_ip', [{'ip-address': '2001:db8::1'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_router.assert_called_once_with(**{'admin_state_up': True, 'name': self.new_router.name, 'external_gateway_info': {'network_id': _network.id, 'enable_snat': True, 'external_fixed_ips': [{'ip_address': '2001:db8::1'}]}})
    self.assertFalse(self.network_client.set_tags.called)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)