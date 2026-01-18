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
@overload(operator.iadd)
def overload_iadd_dummy(arg1, arg2):
    if isinstance(arg1, (MyDummyType, MyDummyType2)) and isinstance(arg2, (MyDummyType, MyDummyType2)):

        def dummy_iadd_impl(arg1, arg2):
            return 42
        return dummy_iadd_impl