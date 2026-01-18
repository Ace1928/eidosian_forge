from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(int)
def class_int(x):
    options = [try_call_method(x, '__int__')]
    options.append(try_call_method(x, '__index__'))
    return take_first(*options)