import unittest
from traits.util.event_tracer import (
def test_change_message_record(self):
    record = ChangeMessageRecord(time=1, indent=3, name='john', old=1, new=1, class_name='MyClass')
    self.assertEqual(str(record), "1 -----> 'john' changed from 1 to 1 in 'MyClass'\n")
    self.assertRaises(TypeError, ChangeMessageRecord, sdd=0)