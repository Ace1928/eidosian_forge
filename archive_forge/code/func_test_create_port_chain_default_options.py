from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_port_chain_default_options(self):
    arglist = [self._port_chain['name'], '--port-pair-group', self._port_chain['port_pair_groups']]
    verifylist = [('name', self._port_chain['name']), ('port_pair_groups', [self._port_chain['port_pair_groups']]), ('flow_classifiers', []), ('chain_parameters', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network.create_sfc_port_chain.assert_called_once_with(**{'name': self._port_chain['name'], 'port_pair_groups': [self._port_chain['port_pair_groups']]})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)