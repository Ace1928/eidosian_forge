import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def getcoroutinestate(coroutine):
    """Get current state of a coroutine object.

    Possible states are:
      CORO_CREATED: Waiting to start execution.
      CORO_RUNNING: Currently being executed by the interpreter.
      CORO_SUSPENDED: Currently suspended at an await expression.
      CORO_CLOSED: Execution has completed.
    """
    if coroutine.cr_running:
        return CORO_RUNNING
    if coroutine.cr_suspended:
        return CORO_SUSPENDED
    if coroutine.cr_frame is None:
        return CORO_CLOSED
    return CORO_CREATED