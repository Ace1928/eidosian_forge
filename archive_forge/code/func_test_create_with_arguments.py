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
def test_create_with_arguments(self):
    m = mock.Mock()

    class Object(object):

        @periodics.periodic(0.5)
        def func1(self, *args, **kwargs):
            m(*args, **kwargs)
    executor_factory = lambda: self.executor_cls(**self.executor_kwargs)
    w = periodics.PeriodicWorker.create(objects=[Object()], executor_factory=executor_factory, args=('foo',), kwargs={'bar': 'baz'}, **self.worker_kwargs)
    with self.create_destroy(w.start):
        self.sleep(2.0)
        w.stop()
    m.assert_called_with('foo', bar='baz')