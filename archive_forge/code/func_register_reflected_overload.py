from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
def register_reflected_overload(func, meth_forward, meth_reflected):

    def class_lt(x, y):
        normal_impl = try_call_method(x, f'__{meth_forward}__', 2)
        if f'__{meth_reflected}__' in y.jit_methods:

            def reflected_impl(x, y):
                return y > x
        else:
            reflected_impl = None
        return take_first(normal_impl, reflected_impl)
    class_instance_overload(func)(class_lt)