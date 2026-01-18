import unittest
from traits.api import (
def test_dict_event_repr(self):
    self.foo.adict.update({'blue': 10, 'black': 0})
    event = self.foo.event
    event_str = "TraitDictEvent(removed={}, added={'black': 0}, changed={'blue': 0})"
    self.assertEqual(repr(event), event_str)
    self.assertIsInstance(eval(repr(event)), TraitDictEvent)