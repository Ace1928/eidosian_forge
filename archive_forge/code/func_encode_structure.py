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
@tf_export('__internal__.saved_model.encode_structure', v1=[])
def encode_structure(nested_structure):
    """Encodes nested structures composed of encodable types into a proto.

  Args:
    nested_structure: Structure to encode.

  Returns:
    Encoded proto.

  Raises:
    NotEncodableError: For values for which there are no encoders.
  """
    return _map_structure(nested_structure, _get_encoders())