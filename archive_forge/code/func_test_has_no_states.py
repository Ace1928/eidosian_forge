from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_has_no_states(self):
    e = events.EventPayload(mock.ANY)
    self.assertFalse(e.has_states)