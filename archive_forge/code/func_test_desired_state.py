from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_desired_state(self):
    desired_state = {'k': object()}
    e = events.DBEventPayload(mock.ANY, desired_state=desired_state)
    self.assertEqual(desired_state, e.desired_state)
    desired_state['a'] = 'A'
    self.assertEqual(desired_state, e.desired_state)