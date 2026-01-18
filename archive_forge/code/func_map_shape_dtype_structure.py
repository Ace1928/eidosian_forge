import json
import shutil
import tempfile
import unittest
import numpy as np
from keras.src import backend
from keras.src import distribution
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils import tree
def map_shape_dtype_structure(fn, shape, dtype):
    """Variant of tree.map_structure that operates on shape tuples."""
    if is_shape_tuple(shape):
        return fn(tuple(shape), dtype)
    if isinstance(shape, list):
        return [map_shape_dtype_structure(fn, s, d) for s, d in zip(shape, dtype)]
    if isinstance(shape, tuple):
        return tuple((map_shape_dtype_structure(fn, s, d) for s, d in zip(shape, dtype)))
    if isinstance(shape, dict):
        return {k: map_shape_dtype_structure(fn, v, dtype[k]) for k, v in shape.items()}
    else:
        raise ValueError(f'Cannot map function to unknown objects {shape} and {dtype}')