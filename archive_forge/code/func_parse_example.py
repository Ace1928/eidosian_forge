from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
from tensorflow.python.ops.gen_parsing_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['io.parse_example', 'parse_example'])
@dispatch.add_dispatch_support
def parse_example(serialized, features, name=None, example_names=None):
    return parse_example_v2(serialized, features, example_names, name)