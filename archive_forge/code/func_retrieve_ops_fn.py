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
def retrieve_ops_fn():
    """Returns the retrieve ops for SGD embedding tables.

      Returns:
        A list of ops to retrieve embedding and slot variables from TPU to CPU.
      """
    retrieve_op_list = []
    config = config_proto
    for host_id, table_variable in enumerate(table_variables):
        with ops.colocate_with(table_variable):
            retrieved_table = tpu_ops.retrieve_tpu_embedding_stochastic_gradient_descent_parameters(table_name=table, num_shards=num_hosts, shard_id=host_id, config=config)
            retrieve_parameters_op = control_flow_ops.group(state_ops.assign(table_variable, retrieved_table))
        config = None
        retrieve_op_list.append(retrieve_parameters_op)
    return retrieve_op_list