from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_list_sfc_service_graphs_with_long_option(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns = self.cmd.take_action(parsed_args)[0]
    sgs = self.network.sfc_service_graphs()
    sg = sgs[0]
    data = [sg['id'], sg['name'], sg['port_chains'], sg['description'], sg['project_id']]
    self.assertEqual(list(self.columns_long), columns)
    self.assertEqual(self.data_long, data)