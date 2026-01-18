from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
def make_overload_template(func, overload_func, jit_options, strict, inline, prefer_literal=False, **kwargs):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    Compiler options are passed as a dictionary to *jit_options*.
    """
    func_name = getattr(func, '__name__', str(func))
    name = 'OverloadTemplate_%s' % (func_name,)
    base = _OverloadFunctionTemplate
    dct = dict(key=func, _overload_func=staticmethod(overload_func), _impl_cache={}, _compiled_overloads={}, _jit_options=jit_options, _strict=strict, _inline=staticmethod(InlineOptions(inline)), _inline_overloads={}, prefer_literal=prefer_literal, metadata=kwargs)
    return type(base)(name, (base,), dct)