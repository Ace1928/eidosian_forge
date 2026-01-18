from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_delete_bgp_speaker(self):
    arglist = [self._bgp_speaker['name']]
    verifylist = [('bgp_speaker', self._bgp_speaker['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.networkclient.delete_bgp_speaker.assert_called_once_with(self._bgp_speaker['name'])
    self.assertIsNone(result)