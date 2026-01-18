from functools import wraps
import inspect
import operator
from numba.core.extending import overload
from numba.core.types import ClassInstanceType
@wraps(extract_template(template, 'func'))
def overload_func(*args, **kwargs):
    options = [try_call_method(args[0], f'__{attr}__', n_args) for attr in attrs]
    return take_first(*options)