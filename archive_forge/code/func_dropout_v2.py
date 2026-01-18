import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(nn_ops.dropout_v2)
def dropout_v2(x: ragged_tensor.Ragged, rate, noise_shape=None, seed=None, name=None):
    """Ragged dispatch target for tf.nn.dropout."""
    if noise_shape is not None:
        raise ValueError('noise_shape is not supported yet for RaggedTensor x')
    with ops.name_scope(name, 'RaggedNNDropout', [x, rate]):
        x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
        return x.with_flat_values(nn_ops.dropout_v2(x.flat_values, rate=rate, seed=seed))