import copy
from unittest import mock
import testscenarios
from heatclient import exc
from heatclient.osc.v1 import event
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import events
@mock.patch('osc_lib.utils.sort_items')
def test_event_list_sort_multiple(self, mock_sort_items):
    arglist = ['my_stack', '--sort', 'resource_name:desc', '--sort', 'id:asc', '--format', 'table']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    mock_event = self.MockEvent()
    mock_sort_items.return_value = [mock_event]
    columns, data = self.cmd.take_action(parsed_args)
    mock_sort_items.assert_called_with(mock.ANY, 'resource_name:desc,id:asc')
    self.event_client.list.assert_called_with(filters={}, resource_name=None, sort_dir='desc', sort_keys=['resource_name', 'id'], stack_id='my_stack')
    self.assertEqual(self.fields, columns)