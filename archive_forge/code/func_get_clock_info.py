import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def get_clock_info():
    """
    Returns a dict containing the OS API used for timing, the precision of the
    underlying clock.
    """
    return _yappi.get_clock_info()