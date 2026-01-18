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
def test_add_on_demand(self):
    called = set()

    def cb(name):
        called.add(name)
    callables = []
    for i in range(0, 10):
        i_cb = functools.partial(cb, '%s_has_called' % i)
        callables.append((every_half_sec, (i_cb,), {}))
    leftover_callables = list(callables)
    w = periodics.PeriodicWorker([], **self.worker_kwargs)
    with self.create_destroy(w.start, allow_empty=True):
        while len(called) != len(callables):
            if leftover_callables:
                cb, args, kwargs = leftover_callables.pop()
                w.add(cb, *args, **kwargs)
            self.sleep(0.1)
        w.stop()