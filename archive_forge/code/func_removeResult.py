import signal
import weakref
from functools import wraps
def removeResult(result):
    return bool(_results.pop(result, None))