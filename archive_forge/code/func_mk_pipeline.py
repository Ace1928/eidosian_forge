import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
@classmethod
def mk_pipeline(cls, args, return_type=None, flags=None, locals={}, library=None, typing_context=None, target_context=None):
    if not flags:
        flags = Flags()
    flags.nrt = True
    if typing_context is None:
        typing_context = cpu_target.typing_context
    if target_context is None:
        target_context = cpu_target.target_context
    return cls(typing_context, target_context, library, args, return_type, flags, locals)