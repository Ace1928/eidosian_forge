import builtins
import torch
from . import _dtypes_impl
def issubclass_(arg, klass):
    try:
        return issubclass(arg, klass)
    except TypeError:
        return False