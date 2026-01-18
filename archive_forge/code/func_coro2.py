import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def coro2():
    evt.wait()
    self.mutex.acquire()
    sequence.append('coro2 acquire')
    self.mutex.release()
    sequence.append('coro2 release')