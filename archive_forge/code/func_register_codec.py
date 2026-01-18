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
def register_codec(x):
    """Registers a codec to use for encoding/decoding.

  Args:
    x: The codec object to register. The object must implement can_encode,
      do_encode, can_decode, and do_decode. See the various _*Codec classes for
      examples.
  """
    _codecs.append(x)