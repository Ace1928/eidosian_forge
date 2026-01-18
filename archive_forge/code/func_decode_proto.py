import collections
import functools
import warnings
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.saved_model.decode_proto', v1=[])
def decode_proto(proto):
    """Decodes proto representing a nested structure.

  Args:
    proto: Proto to decode.

  Returns:
    Decoded structure.

  Raises:
    NotEncodableError: For values for which there are no encoders.
  """
    return _map_structure(proto, _get_decoders())