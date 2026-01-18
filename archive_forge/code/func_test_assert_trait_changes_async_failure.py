import threading
import time
import warnings
from traits.api import (
from traits.testing.api import UnittestTools
from traits.testing.unittest_tools import unittest
from traits.util.api import deprecated
def test_assert_trait_changes_async_failure(self):
    thread_count = 10
    events_per_thread = 10000

    class A(HasTraits):
        event = Event
    a = A()

    def thread_target(obj, count):
        """Fire obj.event 'count' times."""
        for _ in range(count):
            obj.event = True
    threads = [threading.Thread(target=thread_target, args=(a, events_per_thread)) for _ in range(thread_count)]
    expected_count = thread_count * events_per_thread
    with self.assertRaises(AssertionError):
        with self.assertTraitChangesAsync(a, 'event', expected_count + 1):
            for t in threads:
                t.start()
    for t in threads:
        t.join()