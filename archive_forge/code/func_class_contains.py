from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(operator.contains)
def class_contains(x, y):
    return try_call_method(x, '__contains__', 2)