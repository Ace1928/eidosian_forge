import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def pthread1():
    thread_id.append(id(eventlet.greenthread.getcurrent()))
    self.mutex.acquire()
    owner.append(self.mutex.owner)
    pthread2_event1.set()
    orig_os_write = pipe_mutex.os.write

    def patched_os_write(*a, **kw):
        try:
            return orig_os_write(*a, **kw)
        finally:
            pthread1_event.wait()
    with mock.patch.object(pipe_mutex.os, 'write', patched_os_write):
        self.mutex.release()
    pthread2_event2.set()