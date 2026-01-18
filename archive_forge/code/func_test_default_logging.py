import unittest
from unittest import mock
from traits.observation.exception_handling import (
def test_default_logging(self):
    stack = ObserverExceptionHandlerStack()
    with self.assertLogs('traits', level='ERROR') as log_context:
        try:
            raise ZeroDivisionError()
        except Exception:
            stack.handle_exception('Event')
    content, = log_context.output
    self.assertIn('Exception occurred in traits notification handler for event object: {!r}'.format('Event'), content)