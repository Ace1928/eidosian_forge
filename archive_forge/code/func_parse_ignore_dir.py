import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def parse_ignore_dir(s):
    s = os.path.expanduser(os.path.expandvars(s))
    s = s.replace('$prefix', _prefix).replace('$exec_prefix', _exec_prefix)
    return os.path.normpath(s)