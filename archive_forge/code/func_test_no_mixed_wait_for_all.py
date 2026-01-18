import time
import eventlet
import testscenarios
import futurist
from futurist.tests import base
from futurist import waiters
def test_no_mixed_wait_for_all(self):
    fs = [futurist.GreenFuture(), futurist.Future()]
    self.assertRaises(RuntimeError, waiters.wait_for_all, fs)