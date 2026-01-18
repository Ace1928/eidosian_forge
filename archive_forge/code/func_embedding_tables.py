import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@property
def embedding_tables(self) -> Dict[tpu_embedding_v2_utils.TableConfig, tf_variables.Variable]:
    """Returns a dict of embedding tables, keyed by `TableConfig`.

    This property only works when the `TPUEmbedding` object is created under a
    non-TPU strategy. This is intended to be used to for CPU based lookup when
    creating a serving checkpoint.

    Returns:
      A dict of embedding tables, keyed by `TableConfig`.

    Raises:
      RuntimeError: If object was created under a `TPUStrategy`.
    """
    if self._using_tpu:
        if save_context.in_save_context():
            return {table: self._variables[table.name]['parameters'].variables[0] for table in self._table_config}
        raise RuntimeError('Unable to retrieve embedding tables when using a TPU strategy. If you need access, save your model, create this object under a CPU strategy and restore.')
    self._maybe_build(None)
    return {table: self._variables[table.name]['parameters'] for table in self._table_config}