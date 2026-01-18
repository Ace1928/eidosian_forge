from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_options_subnet_ipv6_pd(self):
    self.network_client.create_subnet.return_value = self._subnet_ipv6_pd
    self._network.id = self._subnet_ipv6_pd.network_id
    arglist = [self._subnet_ipv6_pd.name, '--network', self._subnet_ipv6_pd.network_id, '--ip-version', str(self._subnet_ipv6_pd.ip_version), '--ipv6-ra-mode', self._subnet_ipv6_pd.ipv6_ra_mode, '--ipv6-address-mode', self._subnet_ipv6_pd.ipv6_address_mode, '--dhcp', '--use-prefix-delegation']
    verifylist = [('name', self._subnet_ipv6_pd.name), ('network', self._subnet_ipv6_pd.network_id), ('ip_version', self._subnet_ipv6_pd.ip_version), ('ipv6_ra_mode', self._subnet_ipv6_pd.ipv6_ra_mode), ('ipv6_address_mode', self._subnet_ipv6_pd.ipv6_address_mode), ('dhcp', self._subnet_ipv6_pd.enable_dhcp), ('use_prefix_delegation', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet.assert_called_once_with(**{'enable_dhcp': self._subnet_ipv6_pd.enable_dhcp, 'ip_version': self._subnet_ipv6_pd.ip_version, 'ipv6_address_mode': self._subnet_ipv6_pd.ipv6_address_mode, 'ipv6_ra_mode': self._subnet_ipv6_pd.ipv6_ra_mode, 'name': self._subnet_ipv6_pd.name, 'network_id': self._subnet_ipv6_pd.network_id, 'subnetpool_id': self._subnet_ipv6_pd.subnetpool_id})
    self.assertFalse(self.network_client.set_tags.called)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data_ipv6_pd, data)