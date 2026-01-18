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
def can_compile_class(cls) -> bool:
    if is_ignored_fn(cls):
        return False
    ignored_builtin_classes = (torch.nn.Module, tuple, list, Exception)
    if issubclass(cls, ignored_builtin_classes):
        return False
    names = cls.__dict__
    fns = [getattr(cls, name) for name in names if inspect.isroutine(getattr(cls, name, None))]
    has_code = [hasattr(fn, '__code__') for fn in fns]
    return all(has_code)