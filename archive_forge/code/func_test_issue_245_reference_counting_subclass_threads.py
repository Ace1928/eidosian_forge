from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def test_issue_245_reference_counting_subclass_threads(self):
    from threading import Thread
    from threading import Event
    from greenlet import getcurrent

    class MyGreenlet(RawGreenlet):
        pass
    glets = []
    ref_cleared = Event()

    def greenlet_main():
        getcurrent().parent.switch()

    def thread_main(greenlet_running_event):
        mine = MyGreenlet(greenlet_main)
        glets.append(mine)
        mine.switch()
        del mine
        greenlet_running_event.set()
        ref_cleared.wait(10)
        getcurrent()
    initial_refs = sys.getrefcount(MyGreenlet)
    thread_ready_events = []
    for _ in range(initial_refs + 45):
        event = Event()
        thread = Thread(target=thread_main, args=(event,))
        thread_ready_events.append(event)
        thread.start()
    for done_event in thread_ready_events:
        done_event.wait(10)
    del glets[:]
    ref_cleared.set()
    self.wait_for_pending_cleanups()
    self.assertEqual(sys.getrefcount(MyGreenlet), initial_refs)