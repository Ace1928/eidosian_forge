from __future__ import absolute_import, print_function, division
import hashlib
import random as pyrandom
import time
from collections import OrderedDict
from functools import partial
from petl.compat import xrange, text_type
from petl.util.base import Table
def reseed(self):
    self.seed = randomseed()