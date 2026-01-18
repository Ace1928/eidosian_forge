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
def parseNestedExpr(expr, module) -> Tuple[Any, int]:
    i = 0
    while i < len(expr) and expr[i] not in (',', '[', ']'):
        i += 1
    if expr[:i] == '()':
        return ((), i)
    base = lookupInModule(expr[:i].strip(), module)
    assert base is not None, f'Unresolvable type {expr[:i]}'
    if i == len(expr) or expr[i] != '[':
        return (base, i)
    assert expr[i] == '['
    parts = []
    while expr[i] != ']':
        part_len = 0
        i += 1
        part, part_len = parseNestedExpr(expr[i:], module)
        parts.append(part)
        i += part_len
    if len(parts) > 1:
        return (base[tuple(parts)], i + 1)
    else:
        return (base[parts[0]], i + 1)