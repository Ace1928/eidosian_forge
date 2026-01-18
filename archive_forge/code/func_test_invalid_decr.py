import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_invalid_decr(self):
    it = misc.countdown_iter(10, -1)
    self.assertRaises(ValueError, next, it)