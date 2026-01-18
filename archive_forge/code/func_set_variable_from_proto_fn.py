from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
def set_variable_from_proto_fn(variable_from_proto_fn):
    """Set the variable class that variable proto defs will be converted to."""
    global _variable_from_proto_fn
    _variable_from_proto_fn = variable_from_proto_fn