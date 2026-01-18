import signal
import sys
from numba import njit
import numpy as np
def busy_func(a, b, q=None):
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signal.SIGTERM, sigterm_handler)
    try:
        z = busy_func_inner(a, b)
        sys.stdout.flush()
        sys.stderr.flush()
        return z
    except Exception as e:
        if q is not None:
            q.put(e)