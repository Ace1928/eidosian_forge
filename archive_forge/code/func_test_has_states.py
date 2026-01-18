from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_has_states(self):
    e = events.EventPayload(mock.ANY, states=['s1'])
    self.assertTrue(e.has_states)