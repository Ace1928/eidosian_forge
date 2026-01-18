import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def custom_reduce(cls, states):
    """For customizing object serialization in `__reduce__`.

    Object states provided here are used as keyword arguments to the
    `._rebuild()` class method.

    Parameters
    ----------
    states : dict
        Dictionary of object states to be serialized.

    Returns
    -------
    result : tuple
        This tuple conforms to the return type requirement for `__reduce__`.
    """
    return (custom_rebuild, (_CustomPickled(cls, states),))