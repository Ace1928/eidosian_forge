import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_callable_disabled_if_handler_deleted(self):
    dummy = DummyObservable()
    dummy.internal_object = DummyObservable()
    event_factory = mock.Mock()
    notifier = create_notifier(handler=dummy.handler, event_factory=event_factory)
    notifier.add_to(dummy.internal_object)
    notifier(a=1, b=2)
    self.assertEqual(event_factory.call_count, 1)
    event_factory.reset_mock()
    del dummy
    notifier(a=1, b=2)
    event_factory.assert_not_called()