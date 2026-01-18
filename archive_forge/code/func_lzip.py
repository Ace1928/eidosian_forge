import types
import sys
import numbers
import functools
import copy
import inspect
def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs))