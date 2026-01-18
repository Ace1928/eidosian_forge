import types
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.name_scope import name_scope as base_name_scope
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.naming import auto_name
def convert_tf_to_keras_tensor(x):
    if tf.is_tensor(x):
        return KerasTensor(x.shape, x.dtype, sparse=isinstance(x, tf.SparseTensor))
    return x