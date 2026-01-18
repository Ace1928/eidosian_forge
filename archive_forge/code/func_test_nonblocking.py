import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def test_nonblocking(self):
    evt_lock1 = eventlet.event.Event()
    evt_lock2 = eventlet.event.Event()
    evt_unlock = eventlet.event.Event()

    def get_the_lock():
        self.mutex.acquire()
        evt_lock1.send('got the lock')
        evt_lock2.wait()
        self.mutex.release()
        evt_unlock.send('released the lock')
    eventlet.spawn(get_the_lock)
    evt_lock1.wait()
    self.assertFalse(self.mutex.acquire(blocking=False))
    evt_lock2.send('please release the lock')
    evt_unlock.wait()
    self.assertTrue(self.mutex.acquire(blocking=False))