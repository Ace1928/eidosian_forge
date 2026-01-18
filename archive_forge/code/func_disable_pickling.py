import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def disable_pickling(typ):
    """This is called on a type to disable pickling
    """
    NumbaPickler.disabled_types.add(typ)
    return typ