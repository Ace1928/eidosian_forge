import unittest
from traits.util.event_tracer import (
def test_base_message_record(self):
    record = SentinelRecord()
    self.assertEqual(str(record), '\n')
    self.assertRaises(TypeError, SentinelRecord, sdd=0)