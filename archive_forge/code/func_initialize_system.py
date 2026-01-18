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
@tf_export(v1=['tpu.initialize_system'])
def initialize_system(embedding_config: Optional[embedding_pb2.TPUEmbeddingConfiguration]=None, job: Optional[Text]=None, compilation_failure_closes_chips: bool=True, tpu_cancellation_closes_chips: Optional[bool]=None) -> core_types.Tensor:
    """Initializes a distributed TPU system for use with TensorFlow.

  Args:
    embedding_config: If not None, a `TPUEmbeddingConfiguration` proto
      describing the desired configuration of the hardware embedding lookup
      tables. If embedding_config is None, no hardware embeddings can be used.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
    compilation_failure_closes_chips: Set the configuration whether
      we want to close TPU chips when there is a compilation failure.
    tpu_cancellation_closes_chips: Set the configuration whether
      we want to close TPU chips when a TPU execution is cancelled. If the value
      is None, the behavior will be determined by the command line flag
      `tpu_cancellation_closes_chips` for the TPU worker. WARNING: this argument
      only applies to TFRT TPU runtime.
  Returns:
    A serialized `TopologyProto` that describes the TPU system. Note:
      the topology must be evaluated using `Session.run` before it can be used.
  """
    config_string = '' if embedding_config is None else embedding_config.SerializeToString()
    tpu_cancellation_closes_chips_enum = 0
    if tpu_cancellation_closes_chips is not None:
        if tpu_cancellation_closes_chips:
            tpu_cancellation_closes_chips_enum = 1
        else:
            tpu_cancellation_closes_chips_enum = 2
    with ops.device(_tpu_system_device_name(job)):
        topology = tpu_ops.configure_distributed_tpu(compilation_failure_closes_chips=compilation_failure_closes_chips, tpu_cancellation_closes_chips=tpu_cancellation_closes_chips_enum)
        if embedding_config is None:
            return topology
        with ops.control_dependencies([topology]):
            embedding_init = tpu_ops.configure_tpu_embedding(config=config_string)
        with ops.control_dependencies([embedding_init]):
            return array_ops.identity(topology, name='tpu_init_identity')