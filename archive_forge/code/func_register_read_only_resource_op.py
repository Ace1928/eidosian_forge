from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity
def register_read_only_resource_op(op_type):
    """Declares that `op_type` does not update its touched resource."""
    RESOURCE_READ_OPS.add(op_type)