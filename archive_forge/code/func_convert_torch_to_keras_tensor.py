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
def convert_torch_to_keras_tensor(x):
    """Convert `torch.Tensor`s to `KerasTensor`s."""
    if is_tensor(x):
        return KerasTensor(x.shape, standardize_dtype(x.dtype))
    return x