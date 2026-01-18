import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_add_to_observable(self):
    dummy = DummyObservable()
    dummy.notifiers = [str, float]
    notifier = create_notifier()
    notifier.add_to(dummy)
    self.assertEqual(dummy.notifiers, [str, float, notifier])