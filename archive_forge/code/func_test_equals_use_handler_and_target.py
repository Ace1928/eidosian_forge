import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_equals_use_handler_and_target(self):
    handler1 = mock.Mock()
    handler2 = mock.Mock()
    target1 = mock.Mock()
    target2 = mock.Mock()
    dispatcher = dispatch_here
    notifier1 = create_notifier(handler=handler1, target=target1, dispatcher=dispatcher)
    notifier2 = create_notifier(handler=handler1, target=target1, dispatcher=dispatcher)
    notifier3 = create_notifier(handler=handler1, target=target2, dispatcher=dispatcher)
    notifier4 = create_notifier(handler=handler2, target=target1, dispatcher=dispatcher)
    self.assertTrue(notifier1.equals(notifier2), 'The two notifiers should consider each other as equal.')
    self.assertTrue(notifier2.equals(notifier1), 'The two notifiers should consider each other as equal.')
    self.assertFalse(notifier3.equals(notifier1), 'Expected the notifiers to be different because targets are not identical.')
    self.assertFalse(notifier4.equals(notifier1), 'Expected the notifiers to be different because the handlers do not compare equally.')