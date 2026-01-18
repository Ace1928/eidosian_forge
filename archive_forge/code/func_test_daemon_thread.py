import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def test_daemon_thread(self):
    death = threading.Event()
    t = tu.daemon_thread(_spinner, death)
    self.assertTrue(t.daemon)