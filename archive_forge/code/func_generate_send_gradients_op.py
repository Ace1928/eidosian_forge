import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
def generate_send_gradients_op(self, feature_to_gradient_dict, step=None):
    """Send gradient to TPU embedding.

    Args:
      feature_to_gradient_dict: dict mapping feature names to gradient wrt
        activations.
      step: the current global step, used for dynamic learning rate.

    Returns:
      SendTPUEmbeddingGradients Op.

    Raises:
      RuntimeError: If `mode` is not `TRAINING`.
    """
    if self._mode != TRAINING:
        raise RuntimeError('Only in training mode gradients need to be sent to TPU embedding; got mode {}.'.format(self._mode))
    if step is None and self._learning_rate_fn:
        raise ValueError('There are dynamic learning rates but step is None.')
    gradients = []
    for table in self._table_to_features_dict:
        for feature in self._table_to_features_dict[table]:
            gradients.append(feature_to_gradient_dict[feature])
    return tpu_ops.send_tpu_embedding_gradients(inputs=gradients, learning_rates=[math_ops.cast(fn(step), dtype=dtypes.float32) for fn in self._learning_rate_fn], config=self.config_proto.SerializeToString())