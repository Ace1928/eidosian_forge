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
def check_args_exist(target_type) -> None:
    if target_type is List or target_type is list:
        raise_error_container_parameter_missing('List')
    elif target_type is Tuple or target_type is tuple:
        raise_error_container_parameter_missing('Tuple')
    elif target_type is Dict or target_type is dict:
        raise_error_container_parameter_missing('Dict')
    elif target_type is None or target_type is Optional:
        raise_error_container_parameter_missing('Optional')