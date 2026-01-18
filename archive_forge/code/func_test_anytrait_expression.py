import unittest
from traits.api import (
from traits.observation.api import (
def test_anytrait_expression(self):
    obj = HasVariousTraits()
    events = []
    obj.observe(events.append, anytrait())
    obj.foo = 23
    obj.bar = 'on'
    self.assertEqual(len(events), 2)
    foo_event, bar_event = events
    self.assertEqual(foo_event.object, obj)
    self.assertEqual(foo_event.name, 'foo')
    self.assertEqual(foo_event.old, 16)
    self.assertEqual(foo_event.new, 23)
    self.assertEqual(bar_event.object, obj)
    self.assertEqual(bar_event.name, 'bar')
    self.assertEqual(bar_event.old, 'off')
    self.assertEqual(bar_event.new, 'on')