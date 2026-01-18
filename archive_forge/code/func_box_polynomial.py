from numba.extending import (models, register_model, type_callable,
from numba.core import types, cgutils
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning, NumbaValueError
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir
@box(types.PolynomialType)
def box_polynomial(typ, val, c):
    """
    Convert a native polynomial structure to a Polynomial object.
    """
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()
    with ExitStack() as stack:
        polynomial = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
        coef_obj = c.box(typ.coef, polynomial.coef)
        with cgutils.early_exit_if_null(c.builder, stack, coef_obj):
            c.builder.store(fail_obj, ret_ptr)
        domain_obj = c.box(typ.domain, polynomial.domain)
        with cgutils.early_exit_if_null(c.builder, stack, domain_obj):
            c.builder.store(fail_obj, ret_ptr)
        window_obj = c.box(typ.window, polynomial.window)
        with cgutils.early_exit_if_null(c.builder, stack, window_obj):
            c.builder.store(fail_obj, ret_ptr)
        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Polynomial))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(coef_obj)
            c.pyapi.decref(domain_obj)
            c.pyapi.decref(window_obj)
            c.builder.store(fail_obj, ret_ptr)
        if typ.n_args == 1:
            res1 = c.pyapi.call_function_objargs(class_obj, (coef_obj,))
            c.builder.store(res1, ret_ptr)
        else:
            res3 = c.pyapi.call_function_objargs(class_obj, (coef_obj, domain_obj, window_obj))
            c.builder.store(res3, ret_ptr)
        c.pyapi.decref(coef_obj)
        c.pyapi.decref(domain_obj)
        c.pyapi.decref(window_obj)
        c.pyapi.decref(class_obj)
    return c.builder.load(ret_ptr)