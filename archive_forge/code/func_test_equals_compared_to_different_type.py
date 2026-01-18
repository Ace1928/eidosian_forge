import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_equals_compared_to_different_type(self):
    notifier = create_notifier()
    self.assertFalse(notifier.equals(float))