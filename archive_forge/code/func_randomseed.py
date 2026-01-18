from __future__ import absolute_import, print_function, division
import hashlib
import random as pyrandom
import time
from collections import OrderedDict
from functools import partial
from petl.compat import xrange, text_type
from petl.util.base import Table
def randomseed():
    """
    Obtain the hex digest of a sha256 hash of the
    current epoch time in nanoseconds.
    """
    time_ns = str(time.time()).encode()
    hash_time = hashlib.sha256(time_ns).hexdigest()
    return hash_time