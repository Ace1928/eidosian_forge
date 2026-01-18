import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable
def convert_to_list(values, sparse_default_value=None):
    """Convert a TensorLike, CompositeTensor, or ndarray into a Python list."""
    if tf_utils.is_ragged(values):
        if isinstance(values, ragged_tensor.RaggedTensor) and (not context.executing_eagerly()):
            values = backend.get_session(values).run(values)
        values = values.to_list()
    if isinstance(values, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
        if sparse_default_value is None:
            if dtypes.as_dtype(values.values.dtype) == dtypes.string:
                sparse_default_value = ''
            else:
                sparse_default_value = -1
        dense_tensor = sparse_ops.sparse_tensor_to_dense(values, default_value=sparse_default_value)
        values = backend.get_value(dense_tensor)
    if isinstance(values, tensor.Tensor):
        values = backend.get_value(values)
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return values