import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
def test_alive(self):
    with self.executor_cls(**self.executor_kwargs) as executor:
        self.assertTrue(executor.alive)
    self.assertFalse(executor.alive)