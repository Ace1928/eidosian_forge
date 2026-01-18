from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(str)
def class_str(x):
    return take_first(try_call_method(x, '__str__'), lambda x: repr(x))