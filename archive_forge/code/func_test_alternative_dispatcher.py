import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_alternative_dispatcher(self):
    events = []

    def dispatcher(handler, *args):
        event, = args
        events.append(event)

    def event_factory(*args, **kwargs):
        return 'Event'
    notifier = create_notifier(dispatcher=dispatcher, event_factory=event_factory)
    notifier(a=1, b=2)
    self.assertEqual(events, ['Event'])