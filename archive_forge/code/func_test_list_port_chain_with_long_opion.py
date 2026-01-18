from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_port_chain_with_long_opion(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)[0]
    pcs = self.network.sfc_port_chains()
    pc = pcs[0]
    data = [pc['id'], pc['name'], pc['project_id'], pc['port_pair_groups'], pc['flow_classifiers'], pc['chain_parameters'], pc['description']]
    self.assertEqual(list(self.columns_long), columns)
    self.assertEqual(self.data_long, data)