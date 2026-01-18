import collections
from numba.core import types
def make_py_timsort(*args):
    return make_timsort_impl(lambda f: f, *args)