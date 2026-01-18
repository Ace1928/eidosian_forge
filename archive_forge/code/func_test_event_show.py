import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
def test_event_show(self):
    arglist = ['--format', self.format, 'my_stack', 'my_resource', '1234']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.event_client.get.return_value = events.Event(None, self.response)
    self.cmd.take_action(parsed_args)
    self.event_client.get.assert_called_with(**{'stack_id': 'my_stack', 'resource_name': 'my_resource', 'event_id': '1234'})