import contextlib
import os
import ml_dtypes
import numpy as np
import torch
import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.config import floatx
from keras.src.utils.nest import pack_sequence_as
def has_none_shape(x):
    """Check for if a `KerasTensor` has dynamic shape."""
    if isinstance(x, KerasTensor):
        return None in x.shape
    return False