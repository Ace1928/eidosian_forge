import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def create_message(func, overload_func, func_sig, ol_sig):
    msg = []
    s = f"{func} from module '{getattr(func, '__module__')}' has mismatched sig."
    msg.append(s)
    msg.append(f'    - expected: {func_sig}')
    msg.append(f'    -      got: {ol_sig}')
    lineno = inspect.getsourcelines(overload_func)[1]
    tmpsrcfile = inspect.getfile(overload_func)
    srcfile = tmpsrcfile.replace(numba.__path__[0], '')
    msg.append(f'from {srcfile}:{lineno}')
    msgstr = '\n' + '\n'.join(msg)
    return msgstr