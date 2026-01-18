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
def test_interval_checking(self):

    @periodics.periodic(-0.5, enabled=False)
    def no_add_me():
        pass
    w = periodics.PeriodicWorker([], **self.worker_kwargs)
    self.assertEqual(0, len(w))
    self.assertIsNone(w.add(no_add_me))
    self.assertEqual(0, len(w))
    self.assertRaises(ValueError, periodics.periodic, -0.5)