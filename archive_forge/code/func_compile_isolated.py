from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def compile_isolated(pyfunc, argtypes, **kwargs):
    from numba.core.registry import cpu_target
    kwargs.setdefault('return_type', None)
    kwargs.setdefault('locals', {})
    return compile_extra(cpu_target.typing_context, cpu_target.target_context, pyfunc, argtypes, **kwargs)