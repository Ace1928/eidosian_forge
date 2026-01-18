from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.util import object_identity
def get_full_name(var):
    """Gets the full name of variable for name-based checkpoint compatiblity."""
    if not (isinstance(var, variables.Variable) or resource_variable_ops.is_resource_variable(var)):
        return ''
    if getattr(var, '_save_slice_info', None) is not None:
        return var._save_slice_info.full_name
    else:
        return var._shared_name