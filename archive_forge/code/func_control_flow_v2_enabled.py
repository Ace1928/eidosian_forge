from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['control_flow_v2_enabled'])
def control_flow_v2_enabled():
    """Returns `True` if v2 control flow is enabled.

  Note: v2 control flow is always enabled inside of tf.function.
  """
    return control_flow_util.EnableControlFlowV2(ops.get_default_graph())