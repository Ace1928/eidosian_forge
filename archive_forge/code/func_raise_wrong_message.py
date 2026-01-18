import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def raise_wrong_message():
    warnings.warn('foo')