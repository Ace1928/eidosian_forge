from numba.extending import (models, register_model, type_callable,
from numba.core import types, cgutils
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaValueError
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir
@lower_builtin(Polynomial, types.Array)
def impl_polynomial1(context, builder, sig, args):

    def to_double(arr):
        return np.asarray(arr, dtype=np.double)

    def const_impl():
        return np.asarray([-1, 1])
    typ = sig.return_type
    polynomial = cgutils.create_struct_proxy(typ)(context, builder)
    sig_coef = sig.args[0].copy(dtype=types.double)(sig.args[0])
    coef_cast = context.compile_internal(builder, to_double, sig_coef, args)
    sig_domain = sig.args[0].copy(dtype=types.intp)()
    sig_window = sig.args[0].copy(dtype=types.intp)()
    domain_cast = context.compile_internal(builder, const_impl, sig_domain, ())
    window_cast = context.compile_internal(builder, const_impl, sig_window, ())
    polynomial.coef = coef_cast
    polynomial.domain = domain_cast
    polynomial.window = window_cast
    return polynomial._getvalue()