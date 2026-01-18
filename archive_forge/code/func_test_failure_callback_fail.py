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
def test_failure_callback_fail(self):
    worker_kwargs = self.worker_kwargs.copy()
    worker_kwargs['on_failure'] = 'not-a-func'
    self.assertRaises(ValueError, periodics.PeriodicWorker, [], **worker_kwargs)