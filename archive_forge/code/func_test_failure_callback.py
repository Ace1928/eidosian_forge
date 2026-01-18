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
def test_failure_callback(self):
    captures = []
    ev = self.event_cls()

    @periodics.periodic(0.1)
    def failing_always():
        raise RuntimeError('Broke!')

    def trap_failures(cb, kind, periodic_spacing, exc_info, traceback=None):
        captures.append([cb, kind, periodic_spacing, traceback])
        ev.set()
    callables = [(failing_always, None, None)]
    worker_kwargs = self.worker_kwargs.copy()
    worker_kwargs['on_failure'] = trap_failures
    w = periodics.PeriodicWorker(callables, **worker_kwargs)
    with self.create_destroy(w.start, allow_empty=True):
        ev.wait()
        w.stop()
    self.assertEqual(captures[0], [failing_always, periodics.PERIODIC, 0.1, mock.ANY])