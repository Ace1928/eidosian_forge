import inspect
import functools
import sys
import time
def function_timer(func, *args, **kwargs):
    out = kwargs.pop('timeout', sys.stdout)
    t1 = time.time()
    r = func(*args, **kwargs)
    t2 = time.time()
    print(t2 - t1, file=out)
    return r