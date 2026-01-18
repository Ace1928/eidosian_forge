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
@functools.wraps(method_resolver)
def attribute_resolver(self, ty):

    class MethodTemplate(AbstractTemplate):
        key = template_key

        def generic(_, args, kws):
            sig = method_resolver(self, ty, args, kws)
            if sig is not None and sig.recvr is None:
                sig = sig.replace(recvr=ty)
            return sig
    return types.BoundFunction(MethodTemplate, ty)