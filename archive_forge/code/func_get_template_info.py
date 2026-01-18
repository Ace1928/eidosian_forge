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
def get_template_info(self):
    basepath = os.path.dirname(os.path.dirname(numba.__file__))
    impl = self._overload_func
    code, firstlineno, path = self.get_source_code_info(impl)
    sig = str(utils.pysignature(impl))
    info = {'kind': 'overload_method', 'name': getattr(impl, '__qualname__', impl.__name__), 'sig': sig, 'filename': utils.safe_relpath(path, start=basepath), 'lines': (firstlineno, firstlineno + len(code) - 1), 'docstring': impl.__doc__}
    return info