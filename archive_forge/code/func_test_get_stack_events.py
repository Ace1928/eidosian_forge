from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
def test_get_stack_events(self):

    def event_stub(stack_id, argfoo):
        return [self._mock_event('event1', 'aresource')]
    mock_client = mock.MagicMock()
    mock_client.events.list.side_effect = event_stub
    ev_args = {'argfoo': 123}
    evs = event_utils._get_stack_events(hc=mock_client, stack_id='astack/123', event_args=ev_args)
    mock_client.events.list.assert_called_once_with(stack_id='astack/123', argfoo=123)
    self.assertEqual(1, len(evs))
    self.assertEqual('event1', evs[0].id)
    self.assertEqual('astack', evs[0].stack_name)