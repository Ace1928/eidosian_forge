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
def createResolutionCallbackFromFrame(frames_up: int=0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
            print(cb("foo"))

        def baz():
            foo = 2
            bar()

        baz()
    """
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        assert frame is not None
        frame = frame.f_back
        i += 1
    assert frame is not None
    f_locals = frame.f_locals
    f_globals = frame.f_globals

    class env:

        def __getattr__(self, key):
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
            elif key in dir(builtins):
                return getattr(builtins, key)
    return createResolutionCallbackFromEnv(env())