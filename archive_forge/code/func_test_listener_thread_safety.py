import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
def test_listener_thread_safety(self):
    a = A()
    stop_event = threading.Event()
    t = threading.Thread(target=foo_writer, args=(a, stop_event))
    t.start()
    for _ in range(100):
        a.on_trait_change(a.foo_changed_handler, 'foo')
        time.sleep(0.0001)
        a.on_trait_change(a.foo_changed_handler, 'foo', remove=True)
    stop_event.set()
    t.join()
    self.assertTrue(a.exception is None)