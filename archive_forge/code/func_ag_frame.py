import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
@property
def ag_frame(self):
    return self._coroutine.cr_frame