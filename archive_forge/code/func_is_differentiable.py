from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def is_differentiable(op):
    try:
        return ops._gradient_registry.lookup(op.op_def.name) is not None
    except LookupError:
        return False