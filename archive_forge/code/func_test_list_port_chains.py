from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_port_chains(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)[0]
    pcs = self.network.sfc_port_chains()
    pc = pcs[0]
    data = [pc['id'], pc['name'], pc['port_pair_groups'], pc['flow_classifiers'], pc['chain_parameters']]
    self.assertEqual(list(self.columns), columns)
    self.assertEqual(self.data, data)