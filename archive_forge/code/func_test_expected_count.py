import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_expected_count(self):
    upper = 100
    it = misc.countdown_iter(upper)
    items = []
    for i in it:
        self.assertEqual(upper, i)
        upper -= 1
        items.append(i)
    self.assertEqual(0, upper)
    self.assertEqual(100, len(items))