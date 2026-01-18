import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def expanddim_inputs_for_broadcast(self):
    """Reshapes stacked inputs to prepare them for broadcast.

    Since stacked inputs have an extra leading dimension, automatic broadcasting
    rules could incorrectly try to expand dimensions before that leading
    dimension. To avoid that, we reshape these stacked inputs to the maximum
    rank they will need to be broadcasted to.
    """
    if not self._inputs:
        return

    def _get_rank(x):
        rank = array_ops.rank(x.t)
        if not x.is_stacked:
            rank += 1
        return rank
    ranks = [_get_rank(x) for x in self._inputs]
    max_rank = ranks[0]
    for rank in ranks[1:]:
        max_rank = math_ops.maximum(rank, max_rank)
    for i, inp in enumerate(self._inputs):
        if inp.is_stacked:
            shape = array_ops.shape(inp.t)
            rank_diff = array_ops.reshape(max_rank - ranks[i], [1])
            ones = constant_op.constant([1], dtype=shape.dtype)
            ones = array_ops.tile(ones, rank_diff)
            new_shape = array_ops.concat([shape[:1], ones, shape[1:]], axis=0)
            self._inputs[i] = wrap(array_ops.reshape(inp.t, new_shape), True)