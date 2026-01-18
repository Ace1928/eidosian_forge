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
def get_thread_stats():
    """
    Gets the thread profiler results with given filters and returns an iterable.
    """
    return YThreadStats().get()