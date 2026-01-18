import os
import tempfile
import eventlet
from eventlet import greenpool
from oslotest import base as test_base
from oslo_concurrency import lockutils
def test_lock_with_spawn_n(self):
    self._test_internal_lock_with_two_threads(fair=False, spawn=eventlet.spawn_n)