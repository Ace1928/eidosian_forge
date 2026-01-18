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
def sig_from_np_random(x):
    if not x.startswith('_'):
        thing = getattr(np.random, x)
        if inspect.isbuiltin(thing):
            docstr = thing.__doc__.splitlines()
            for l in docstr:
                if l:
                    sl = l.strip()
                    if sl.startswith(x):
                        if x == 'seed':
                            sl = 'seed(seed)'
                        fake_impl = f'def {sl}:\n\tpass'
                        l = {}
                        try:
                            exec(fake_impl, {}, l)
                        except SyntaxError:
                            if DEBUG == 2:
                                print('... skipped as cannot parse signature')
                            return None
                        else:
                            fn = l.get(x)
                            return inspect.signature(fn)