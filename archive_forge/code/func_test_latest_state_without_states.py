from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_latest_state_without_states(self):
    body = object()
    e = events.EventPayload(mock.ANY, request_body=body)
    self.assertIsNone(e.latest_state)