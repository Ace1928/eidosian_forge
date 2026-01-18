import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
@staticmethod
def default_local_init_op():
    """Returns an op that groups the default local init ops.

    This op is used during session initialization when a Scaffold is
    initialized without specifying the local_init_op arg. It includes
    `tf.compat.v1.local_variables_initializer`,
    `tf.compat.v1.tables_initializer`, and also
    initializes local session resources.

    Returns:
      The default Scaffold local init op.
    """
    return control_flow_ops.group(variables.local_variables_initializer(), lookup_ops.tables_initializer(), resources.initialize_resources(resources.local_resources()))