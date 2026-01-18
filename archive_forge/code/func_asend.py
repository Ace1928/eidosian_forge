import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
def asend(self, value):
    return self._do_it(self._it.send, value)