import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
def test_listener_deleted_race(self):

    class SlowListener(HasTraits):

        def handle_age_change(self):
            time.sleep(1.0)

    def worker_thread(event_source, start_event):
        start_event.wait()
        event_source.age = 11

    def main_thread(event_source, start_event):
        listener = SlowListener()
        event_source.on_trait_change(listener.handle_age_change, 'age')
        start_event.set()
        time.sleep(0.5)
        event_source.on_trait_change(listener.handle_age_change, 'age', remove=True)
    with captured_stderr() as s:
        start_event = threading.Event()
        event_source = GenerateEvents(age=10)
        t = threading.Thread(target=worker_thread, args=(event_source, start_event))
        t.start()
        main_thread(event_source, start_event)
        t.join()
    self.assertNotIn('Exception', s.getvalue())