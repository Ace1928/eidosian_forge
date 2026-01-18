import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
def test_event_list_log_format(self):
    arglist = ['my_stack']
    expected = '2015-11-13 10:02:17 [resource1]: CREATE_COMPLETE  state changed\n'
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.run(parsed_args)
    self.event_client.list.assert_called_with(**self.defaults)
    self.assertEqual(expected, self.fake_stdout.make_string())