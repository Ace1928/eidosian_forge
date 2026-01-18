import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
def test_event_set_clear_timeout(self):
    event = eventletutils.EventletEvent()
    wakes = []

    def thread_func():
        result = event.wait(0.2)
        wakes.append(result)
        if len(wakes) == 1:
            self.assertTrue(result)
            event.clear()
        else:
            self.assertFalse(result)
    a = greenthread.spawn(thread_func)
    b = greenthread.spawn(thread_func)
    eventlet.sleep(0)
    event.set()
    with eventlet.timeout.Timeout(0.3):
        a.wait()
        b.wait()
    self.assertFalse(event.is_set())
    self.assertEqual([True, False], wakes)