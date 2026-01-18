from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@property
def supports_serialize(self):
    """Whether the Reader implementation can serialize its state."""
    return self._supports_serialize