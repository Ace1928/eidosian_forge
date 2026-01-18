from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_tap_enabled_port_pair_group(self):
    arglist = ['--description', self._port_pair_group['description'], '--port-pair', self._port_pair_group['port_pairs'], self._port_pair_group['name'], '--enable-tap']
    verifylist = [('port_pairs', [self._port_pair_group['port_pairs']]), ('name', self._port_pair_group['name']), ('description', self._port_pair_group['description']), ('enable_tap', True)]
    expected_data = self._update_expected_response_data(data={'tap_enabled': True})
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network.create_sfc_port_pair_group.assert_called_once_with(**{'name': self._port_pair_group['name'], 'port_pairs': [self._port_pair_group['port_pairs']], 'description': self._port_pair_group['description'], 'tap_enabled': True})
    self.assertEqual(self.columns, columns)
    self.assertEqual(expected_data, data)