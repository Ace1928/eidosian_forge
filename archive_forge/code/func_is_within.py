from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
def is_within(op):
    return (within_ops is None or op in within_ops) and (within_ops_fn is None or within_ops_fn(op))