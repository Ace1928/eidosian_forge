import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def rewrite_for_inference(computation: Callable[..., Any], inputs: Optional[List[core_types.Tensor]]=None, infeed_queue: Optional[tpu_feed.InfeedQueue]=None, device_assignment: Optional[device_assignment_lib.DeviceAssignment]=None, name: Optional[Text]=None) -> List[core_types.Tensor]:
    """Rewrites `computation` for inference on a TPU system.

     Other than 'rewriting' the computation to run on a TPU, if using variables
     in your computation, it moves the ReadVariableOps outside the TPU
     computation, and adds GuaranteeConst ops just after the ReadVariableOps.
     This mechanism works only if you are using tf.compat.v1.get_variable() to
     create and access variables in your tpu computation. You can validate
     whether this worked, by calling validate_inference_rewrite_for_variables()
     method immediately after this method to check whether GuaranteeConstOps
     where added to the graph.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors. If the function returns m outputs, rewrite will return a list of
      m tensors.
    inputs: A list of input tensors or `None` (equivalent to an empty list).
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: if not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. May be omitted for a single-core computation, in which
      case the core attached to task 0, TPU device 0 is used.
    name: The name of the operator.
  Returns:
    A list of output tensors.
  """

    def guarantee_const_getter(getter, name, *args, **kwargs):
        with ops.control_dependencies(None):
            return array_ops.guarantee_const(getter(name, *args, **kwargs), name=name + '/GuaranteeConst')

    def wrapped_computation(*args, **kwargs):
        """Execute computation under `_TPUInferenceContext`."""
        context = _TPUInferenceContext(name=ops.get_default_graph().unique_name('rewrite_for_inference'))
        try:
            context.Enter()
            vscope = variable_scope.get_variable_scope()
            prev_custom_getter = vscope.custom_getter
            prev_caching_device = vscope.caching_device
            vscope.set_custom_getter(guarantee_const_getter)
            vscope.set_caching_device(lambda op: op.device)
            result = computation(*args, **kwargs)
            vscope.set_custom_getter(prev_custom_getter)
            vscope.set_caching_device(prev_caching_device)
        finally:
            context.Exit()
        return result
    return rewrite(wrapped_computation, inputs=inputs, infeed_queue=infeed_queue, device_assignment=device_assignment, name=name)