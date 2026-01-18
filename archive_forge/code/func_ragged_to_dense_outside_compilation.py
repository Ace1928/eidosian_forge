from typing import Any, Dict, Iterable, Optional, Text, Union
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def ragged_to_dense_outside_compilation(inp, weight, batch_size, feature):
    if weight is None:
        weight = ragged_tensor.RaggedTensor.from_row_splits(array_ops.ones_like(inp.values, dtype=dtypes.float32), inp.row_splits)
    if not feature.output_shape and feature.max_sequence_length > 0:
        inp = inp.to_tensor(shape=(batch_size, feature.max_sequence_length))
        weight = array_ops.ones_like(inp, dtype=dtypes.float32)
    elif feature.output_shape:
        with ops.init_scope():
            output_batch_size = math_ops.reduce_prod(feature.output_shape).numpy()
        if output_batch_size == batch_size:
            inp, weight = (inp.to_tensor(), weight.to_tensor())
        elif output_batch_size > batch_size and output_batch_size % batch_size == 0:
            seq_length = output_batch_size // batch_size
            inp = inp.to_tensor(shape=(batch_size, seq_length))
            weight = array_ops.ones_like(inp, dtype=dtypes.float32)
        else:
            raise ValueError('Output shape set in the FeatureConfig should be the factor of the input data batch size. But instead got output shape {}, input data batch size {}'.format(feature.output_shape, batch_size))
    else:
        inp, weight = (inp.to_tensor(), weight.to_tensor())
    return (inp, weight)