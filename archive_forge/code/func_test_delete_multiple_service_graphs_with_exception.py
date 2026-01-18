from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_delete_multiple_service_graphs_with_exception(self):
    client = self.app.client_manager.network
    target = self._service_graph[0]['id']
    arglist = [target]
    verifylist = [('service_graph', [target])]
    client.find_sfc_service_graph.side_effect = [target, exceptions.CommandError]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    msg = '1 of 2 service graph(s) failed to delete.'
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual(msg, str(e))