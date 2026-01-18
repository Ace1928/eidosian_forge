import unittest
from traits.api import (
from traits.observation.api import (
def test_default_not_called_if_init_contains_value(self):
    record = Record(number=123)
    self.assertEqual(record.default_call_count, 1)
    self.assertEqual(len(record.number_change_events), 1)
    event, = record.number_change_events
    self.assertEqual(event.object, record)
    self.assertEqual(event.name, 'number')
    self.assertEqual(event.old, 99)
    self.assertEqual(event.new, 123)