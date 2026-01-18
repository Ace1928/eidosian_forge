import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_no_count(self):
    it = misc.countdown_iter(0)
    self.assertEqual(0, len(list(it)))
    it = misc.countdown_iter(-1)
    self.assertEqual(0, len(list(it)))