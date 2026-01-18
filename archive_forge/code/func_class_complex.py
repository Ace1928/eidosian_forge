from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(complex)
def class_complex(real=0, imag=0):
    return take_first(try_call_complex_method(real, '__complex__'), lambda real=0, imag=0: complex(float(real)))