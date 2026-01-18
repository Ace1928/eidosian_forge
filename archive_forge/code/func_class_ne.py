from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@class_instance_overload(operator.ne)
def class_ne(x, y):
    return take_first(try_call_method(x, '__ne__', 2), lambda x, y: not x == y)