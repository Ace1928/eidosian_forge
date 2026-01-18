import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
def createResolutionCallbackForClassMethods(cls):
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    fns = [getattr(cls, name) for name in cls.__dict__ if inspect.isroutine(getattr(cls, name))]
    fns = [fn for fn in fns if not inspect.isbuiltin(fn) and hasattr(fn, '__globals__')]
    captures = {}
    for fn in fns:
        captures.update(get_closure(fn))
        captures.update(get_type_hint_captures(fn))

    def lookup_in_class(key):
        if key in captures:
            return captures[key]
        else:
            return getattr(builtins, key, None)
    return lookup_in_class