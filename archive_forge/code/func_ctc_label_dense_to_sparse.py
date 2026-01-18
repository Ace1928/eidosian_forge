import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ctc_label_dense_to_sparse(labels, label_lengths):
    """Converts CTC labels from dense to sparse.

  Args:
      labels: dense CTC labels.
      label_lengths: length of the labels.

  Returns:
      A sparse tensor representation of the labels.
  """
    label_shape = array_ops.shape(labels)
    num_batches_tns = array_ops_stack.stack([label_shape[0]])
    max_num_labels_tns = array_ops_stack.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return array_ops.expand_dims(math_ops.range(array_ops.shape(old_input)[1]), 0) < array_ops.fill(max_num_labels_tns, current_input)
    init = math_ops.cast(array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
    dense_mask = functional_ops.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]
    label_array = array_ops.reshape(array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns), label_shape)
    label_ind = array_ops.boolean_mask(label_array, dense_mask)
    batch_array = array_ops.transpose(array_ops.reshape(array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns), reverse(label_shape, 0)))
    batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
    indices = array_ops.transpose(array_ops.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))
    vals_sparse = array_ops.gather_nd(labels, indices)
    return sparse_tensor.SparseTensor(math_ops.cast(indices, dtypes_module.int64), vals_sparse, math_ops.cast(label_shape, dtypes_module.int64))