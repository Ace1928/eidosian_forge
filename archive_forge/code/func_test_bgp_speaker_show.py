from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_speaker
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
def test_bgp_speaker_show(self):
    arglist = [self._bgp_speaker_name]
    verifylist = [('bgp_speaker', self._bgp_speaker_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    data = self.cmd.take_action(parsed_args)
    self.networkclient.get_bgp_speaker.assert_called_once_with(self._bgp_speaker_name)
    self.assertEqual(self.columns, data[0])
    self.assertEqual(self.data, data[1])