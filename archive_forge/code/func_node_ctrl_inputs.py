from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
@property
def node_ctrl_inputs(self):
    return self._node_ctrl_inputs