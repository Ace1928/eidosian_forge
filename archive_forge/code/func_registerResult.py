import signal
import weakref
from functools import wraps
def registerResult(result):
    _results[result] = 1