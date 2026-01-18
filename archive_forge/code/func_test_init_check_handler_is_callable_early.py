import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_init_check_handler_is_callable_early(self):
    not_a_callable = None
    with self.assertRaises(ValueError) as exception_cm:
        create_notifier(handler=not_a_callable)
    self.assertEqual(str(exception_cm.exception), 'handler must be a callable, got {!r}'.format(not_a_callable))