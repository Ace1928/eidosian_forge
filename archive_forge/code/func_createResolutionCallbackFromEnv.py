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
def createResolutionCallbackFromEnv(lookup_base):
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """

    def lookupInModule(qualified_name, module):
        if '.' in qualified_name:
            parts = qualified_name.split('.')
            base = parts[0]
            remaining_pieces = '.'.join(parts[1:])
            module_value = getattr(module, base)
            return lookupInModule(remaining_pieces, module_value)
        else:
            return getattr(module, qualified_name)

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

    def parseExpr(expr, module):
        try:
            value, len_parsed = parseNestedExpr(expr, module)
            assert len_parsed == len(expr), 'whole expression was not parsed, falling back to c++ parser'
            return value
        except Exception:
            '\n            The python resolver fails in several cases in known unit tests, and is intended\n            to fall back gracefully to the c++ resolver in general.  For example, python 2 style\n            annotations which are frequent in our unit tests often fail with types e.g. int not\n            resolvable from the calling frame.\n            '
            return None
    return lambda expr: parseExpr(expr, lookup_base)