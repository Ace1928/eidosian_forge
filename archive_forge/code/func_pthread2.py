import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def pthread2():
    pthread2_event1.wait()
    thread_id.append(id(eventlet.greenthread.getcurrent()))
    self.mutex.acquire()
    pthread1_event.set()
    pthread2_event2.wait()
    owner.append(self.mutex.owner)
    self.mutex.release()