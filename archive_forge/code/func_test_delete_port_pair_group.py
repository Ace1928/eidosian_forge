from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair_group
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_delete_port_pair_group(self):
    client = self.app.client_manager.network
    mock_port_pair_group_delete = client.delete_sfc_port_pair_group
    arglist = [self._port_pair_group[0]['id']]
    verifylist = [('port_pair_group', [self._port_pair_group[0]['id']])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    mock_port_pair_group_delete.assert_called_once_with(self._port_pair_group[0]['id'])
    self.assertIsNone(result)