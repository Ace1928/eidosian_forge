from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_from_subnet_pool_options(self):
    self.network_client.create_subnet.return_value = self._subnet_from_pool
    self.network_client.set_tags = mock.Mock(return_value=None)
    self._network.id = self._subnet_from_pool.network_id
    arglist = [self._subnet_from_pool.name, '--subnet-pool', self._subnet_from_pool.subnetpool_id, '--prefix-length', '24', '--network', self._subnet_from_pool.network_id, '--ip-version', str(self._subnet_from_pool.ip_version), '--gateway', self._subnet_from_pool.gateway_ip, '--dhcp']
    for dns_addr in self._subnet_from_pool.dns_nameservers:
        arglist.append('--dns-nameserver')
        arglist.append(dns_addr)
    for host_route in self._subnet_from_pool.host_routes:
        arglist.append('--host-route')
        value = 'gateway=' + host_route.get('nexthop', '') + ',destination=' + host_route.get('destination', '')
        arglist.append(value)
    for service_type in self._subnet_from_pool.service_types:
        arglist.append('--service-type')
        arglist.append(service_type)
    verifylist = [('name', self._subnet_from_pool.name), ('prefix_length', '24'), ('network', self._subnet_from_pool.network_id), ('ip_version', self._subnet_from_pool.ip_version), ('gateway', self._subnet_from_pool.gateway_ip), ('dns_nameservers', self._subnet_from_pool.dns_nameservers), ('dhcp', self._subnet_from_pool.enable_dhcp), ('host_routes', subnet_v2.convert_entries_to_gateway(self._subnet_from_pool.host_routes)), ('subnet_pool', self._subnet_from_pool.subnetpool_id), ('service_types', self._subnet_from_pool.service_types)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet.assert_called_once_with(**{'dns_nameservers': self._subnet_from_pool.dns_nameservers, 'enable_dhcp': self._subnet_from_pool.enable_dhcp, 'gateway_ip': self._subnet_from_pool.gateway_ip, 'host_routes': self._subnet_from_pool.host_routes, 'ip_version': self._subnet_from_pool.ip_version, 'name': self._subnet_from_pool.name, 'network_id': self._subnet_from_pool.network_id, 'prefixlen': '24', 'subnetpool_id': self._subnet_from_pool.subnetpool_id, 'service_types': self._subnet_from_pool.service_types})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data_subnet_pool, data)