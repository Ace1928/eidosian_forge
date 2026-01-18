from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_port_pairs(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)[0]
    port_pairs = self.network.sfc_port_pairs()
    port_pair = port_pairs[0]
    data = [port_pair['id'], port_pair['name'], port_pair['ingress'], port_pair['egress']]
    self.assertEqual(list(self.columns), columns)
    self.assertEqual(self.data, data)