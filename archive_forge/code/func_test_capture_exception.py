import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_capture_exception(self):
    with self.assertRaises(IndexError):
        pop_exception_handler()

    def misbehaving_handler(event):
        raise ZeroDivisionError('lalalala')
    notifier = create_notifier(handler=misbehaving_handler)
    with self.assertLogs('traits', level='ERROR') as log_exception:
        notifier(a=1, b=2)
    content, = log_exception.output
    self.assertIn('Exception occurred in traits notification handler', content)
    self.assertIn('ZeroDivisionError', content)