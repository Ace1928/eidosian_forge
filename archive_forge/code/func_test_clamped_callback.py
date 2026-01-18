import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_clamped_callback(self):
    calls = []

    def on_clamped():
        calls.append(True)
    misc.clamp(-1, 0.0, 1.0, on_clamped=on_clamped)
    self.assertEqual(1, len(calls))
    calls.pop()
    misc.clamp(0.0, 0.0, 1.0, on_clamped=on_clamped)
    self.assertEqual(0, len(calls))
    misc.clamp(2, 0.0, 1.0, on_clamped=on_clamped)
    self.assertEqual(1, len(calls))