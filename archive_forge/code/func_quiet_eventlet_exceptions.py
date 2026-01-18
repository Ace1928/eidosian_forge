import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
@contextlib.contextmanager
def quiet_eventlet_exceptions():
    orig_state = greenpool.DEBUG
    eventlet_debug.hub_exceptions(False)
    try:
        yield
    finally:
        eventlet_debug.hub_exceptions(orig_state)