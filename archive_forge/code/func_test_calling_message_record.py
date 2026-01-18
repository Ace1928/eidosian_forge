import unittest
from traits.util.event_tracer import (
def test_calling_message_record(self):
    record = CallingMessageRecord(time=7, indent=5, handler='john', source='sssss')
    self.assertEqual(str(record), "7             CALLING: 'john' in sssss\n")
    self.assertRaises(TypeError, CallingMessageRecord, sdd=0)