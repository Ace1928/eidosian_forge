from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_sfc_service_graph_invalid_port_chains(self):
    bp1_str = 'pc1:pc2,pc3:'
    self.cmd = sfc_service_graph.CreateSfcServiceGraph(self.app, self.namespace)
    arglist = ['--description', self._service_graph['description'], '--branching-point', bp1_str, self._service_graph['name']]
    verifylist = [('description', self._service_graph['description']), ('branching_points', [bp1_str]), ('name', self._service_graph['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)