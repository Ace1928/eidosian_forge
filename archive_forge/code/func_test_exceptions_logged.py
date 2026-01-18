import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
def test_exceptions_logged(self):
    ge = GenerateFailingEvents()
    traits_logger = logging.getLogger('traits')
    with self.assertLogs(logger=traits_logger, level=logging.ERROR) as log_watcher:
        ge.name = 'Terry Jones'
    self.assertEqual(len(log_watcher.records), 1)
    log_record = log_watcher.records[0]
    self.assertIn('Exception occurred in traits notification handler', log_record.message)
    _, exc_value, exc_traceback = log_record.exc_info
    self.assertIsInstance(exc_value, RuntimeError)
    self.assertIsNotNone(exc_traceback)