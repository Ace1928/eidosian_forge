import queue
from tensorflow.python.framework import dtypes
def capture_by_value(op):
    return op.outputs[0].dtype in TENSOR_TYPES_ALLOWLIST or op.type in OP_TYPES_ALLOWLIST