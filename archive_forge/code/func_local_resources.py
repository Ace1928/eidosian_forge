import collections
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_should_use
def local_resources():
    """Returns resources intended to be local to this session."""
    return ops.get_collection(ops.GraphKeys.LOCAL_RESOURCES)