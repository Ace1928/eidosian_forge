import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def get_asyncgen_hooks():
    return asyncgen_hooks(firstiter=_hooks.firstiter, finalizer=_hooks.finalizer)