import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_callable_disabled_if_target_removed(self):
    target = mock.Mock()
    handler = mock.Mock()
    notifier = create_notifier(handler=handler, target=target)
    notifier(a=1, b=2)
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    del target
    notifier(a=1, b=2)
    handler.assert_not_called()