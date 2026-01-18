import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
@mock.patch('oslo_utils.eventletutils._eventlet.event.Event')
def test_event_clear_already_sent(self, mock_event):
    old_event = mock.Mock()
    new_event = mock.Mock()
    mock_event.side_effect = [old_event, new_event]
    event = eventletutils.EventletEvent()
    event.set()
    event.clear()
    self.assertEqual(1, old_event.send.call_count)