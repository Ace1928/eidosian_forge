import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def patched_os_write(*a, **kw):
    try:
        return orig_os_write(*a, **kw)
    finally:
        pthread1_event.wait()