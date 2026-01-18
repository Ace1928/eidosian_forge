from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_is_not_persisted(self):
    e = events.DBEventPayload(mock.ANY, states=['s1'])
    self.assertFalse(e.is_persisted)
    e = events.DBEventPayload(mock.ANY, resource_id='1a')
    self.assertFalse(e.is_persisted)