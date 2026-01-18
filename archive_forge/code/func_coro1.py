import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def coro1():
    eventlet.sleep(0)
    self.mutex.acquire()
    sequence.append('coro1 acquire')
    evt.send('go')
    self.mutex.release()
    sequence.append('coro1 release')