import collections
import inspect
import random
import string
import time
import testscenarios
from taskflow import test
from taskflow.utils import misc
from taskflow.utils import threading_utils
def test_mutated_post_copy(self):
    a = {'a': 'b'}
    a_2 = misc.safe_copy_dict(a)
    a['a'] = 'c'
    self.assertEqual('b', a_2['a'])
    self.assertEqual('c', a['a'])