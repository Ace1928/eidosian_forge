import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def get_the_lock():
    self.mutex.acquire()
    evt_lock1.send('got the lock')
    evt_lock2.wait()
    self.mutex.release()
    evt_unlock.send('released the lock')