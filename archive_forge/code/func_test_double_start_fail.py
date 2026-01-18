import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def test_double_start_fail(self):
    w = periodics.PeriodicWorker([], **self.worker_kwargs)
    with self.create_destroy(w.start, allow_empty=True):
        self.sleep(0.5)
        self.assertRaises(RuntimeError, w.start)
        w.stop()