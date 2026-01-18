import functools
import dill
import sys
@functools.lru_cache(None)
def function_with_cache(x):
    global globalvar
    globalvar += x
    return globalvar