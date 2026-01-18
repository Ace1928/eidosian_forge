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
def test_mk_func_literal(self):
    """make sure make_function is passed to typer class as a literal
        """
    test_ir = compiler.run_frontend(mk_func_test_impl)
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    typingctx.refresh()
    targetctx.refresh()
    typing_res = type_inference_stage(typingctx, targetctx, test_ir, (), None)
    self.assertTrue(any((isinstance(a, types.MakeFunctionLiteral) for a in typing_res.typemap.values())))