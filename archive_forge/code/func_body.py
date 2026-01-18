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
def body(not_all_done, indices, *args):
    num_inputs = self._pfor_input.num_inputs
    inputs = args[:num_inputs]
    output_tas = args[num_inputs:]
    inputs_stacked = [x.is_stacked for x in self._pfor_input.inputs]
    assert len(inputs) >= len(output_tas)
    assert len(inputs) == len(inputs_stacked)
    with ops.name_scope('while_cond'):
        cond_pfor = PFor(loop_var=self._pfor.loop_var, loop_len=array_ops.size(indices), pfor_ops=self._cond_func.graph.get_operations(), fallback_to_while_loop=self._pfor.fallback_to_while_loop, all_indices=indices, all_indices_partitioned=True, pfor_config=self._pfor.pfor_config)
        wrapped_inputs = [wrap(inp, stacked) for inp, stacked in zip(inputs, inputs_stacked)]
        conditions, cond_stacked, _ = _convert_function_call(self._cond_func, cond_pfor, wrapped_inputs)[0]
        cond_is_stacked[0] = cond_stacked
    if not cond_stacked:
        not_all_done, new_indices, new_inputs, new_output_tas = self._process_cond_unstacked(conditions, indices, inputs, output_tas)
    else:
        not_all_done, new_indices, new_inputs, new_output_tas = self._process_cond_stacked(conditions, indices, inputs, inputs_stacked, output_tas)
    with ops.name_scope('while_body'):
        new_outputs, mismatching_stacked_indices = self._process_body(inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done)
    indices_to_stack[:] = mismatching_stacked_indices
    for i, new_output in enumerate(new_outputs):
        new_output.set_shape(output_shapes[i])
    new_args = [not_all_done, new_indices] + new_outputs + list(new_output_tas)
    return tuple(new_args)