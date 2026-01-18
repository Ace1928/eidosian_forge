import collections
import collections.abc
import enum
import typing
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations
@staticmethod
def is_reserved_name(name):
    """Returns true if `name` is a reserved name."""
    return name in RESERVED_FIELD_NAMES or name.lower().startswith('_tf_extension_type')