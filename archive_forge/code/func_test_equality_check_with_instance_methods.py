import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_equality_check_with_instance_methods(self):
    instance = DummyObservable()
    target = mock.Mock()
    notifier1 = create_notifier(handler=instance.handler, target=target)
    notifier2 = create_notifier(handler=instance.handler, target=target)
    self.assertTrue(notifier1.equals(notifier2))
    self.assertTrue(notifier2.equals(notifier1))