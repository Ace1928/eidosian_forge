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
def get_class_name_lineno(method) -> Tuple[str, int]:
    current_frame = inspect.currentframe()
    for i in range(2):
        assert current_frame is not None
        current_frame = current_frame.f_back
    assert current_frame is not None
    class_name = current_frame.f_code.co_name
    line_no = current_frame.f_code.co_firstlineno
    return (class_name, line_no)