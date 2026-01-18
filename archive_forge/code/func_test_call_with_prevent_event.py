import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_call_with_prevent_event(self):
    observer_handler = mock.Mock()
    handler = mock.Mock()
    target = mock.Mock()
    notifier = create_notifier(observer_handler=observer_handler, handler=handler, target=target, event_factory=lambda value: value, prevent_event=lambda event: event != 'Fire')
    notifier('Hello')
    self.assertEqual(observer_handler.call_count, 0)
    notifier('Fire')
    self.assertEqual(observer_handler.call_count, 1)