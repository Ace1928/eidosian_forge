import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
@mock.patch('time.sleep')
def test_event_list_follow(self, sleep):
    sleep.side_effect = [None, KeyboardInterrupt()]
    arglist = ['--follow', 'my_stack']
    expected = '2015-11-13 10:02:17 [resource1]: CREATE_COMPLETE  state changed\n2015-11-13 10:02:17 [resource1]: CREATE_COMPLETE  state changed\n'
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    defaults_with_marker = dict(self.defaults)
    defaults_with_marker['marker'] = '1234'
    self.event_client.list.assert_has_calls([mock.call(**self.defaults), mock.call(**defaults_with_marker)])
    self.assertEqual([], columns)
    self.assertEqual([], data)
    self.assertEqual(expected, self.fake_stdout.make_string())