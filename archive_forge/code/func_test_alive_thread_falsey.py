import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def test_alive_thread_falsey(self):
    for v in [False, 0, None, '']:
        self.assertFalse(tu.is_alive(v))