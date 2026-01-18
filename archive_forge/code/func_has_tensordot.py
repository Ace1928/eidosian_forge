import importlib
import numpy
from . import object_arrays
from . import cupy as _cupy
from . import jax as _jax
from . import tensorflow as _tensorflow
from . import theano as _theano
from . import torch as _torch
def has_tensordot(backend):
    """Check if ``{backend}.tensordot`` exists, cache result for performance.
    """
    try:
        return _has_tensordot[backend]
    except KeyError:
        try:
            get_func('tensordot', backend)
            _has_tensordot[backend] = True
        except AttributeError:
            _has_tensordot[backend] = False
        return _has_tensordot[backend]