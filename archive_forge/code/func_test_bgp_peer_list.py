from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_peer
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_bgp_peer_list(self):
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.networkclient.bgp_peers.assert_called_once_with(retrieve_all=True)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))