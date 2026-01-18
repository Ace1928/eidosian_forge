from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_is_to_be_committed(self):
    e = events.DBEventPayload(mock.ANY, states=[mock.ANY], resource_id='1a', desired_state=object())
    self.assertTrue(e.is_to_be_committed)