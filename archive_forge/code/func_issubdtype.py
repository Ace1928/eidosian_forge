import builtins
import torch
from . import _dtypes_impl
def issubdtype(arg1, arg2):
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)