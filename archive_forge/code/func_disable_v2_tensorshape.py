import functools
import operator
from typing import Optional, Sequence, Type
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export(v1=['disable_v2_tensorshape'])
def disable_v2_tensorshape():
    """Disables the V2 TensorShape behavior and reverts to V1 behavior.

  See docstring for `enable_v2_tensorshape` for details about the new behavior.
  """
    global _TENSORSHAPE_V2_OVERRIDE
    _TENSORSHAPE_V2_OVERRIDE = False
    logging.vlog(1, 'Disabling v2 tensorshape')
    _api_usage_gauge.get_cell().set(False)