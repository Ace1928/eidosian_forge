import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
def test_blows_up(self):
    fut = self.executor.submit(blows_up)
    self.assertRaises(RuntimeError, fut.result)
    self.assertIsInstance(fut.exception(), RuntimeError)