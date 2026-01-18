import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings
import numpy as np
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def batchable_from_tensor_list(spec, tensor_list):
    """Returns a value with type `spec` decoded from `tensor_list`."""
    if isinstance(spec, internal.TensorSpec):
        assert len(tensor_list) == 1
        return tensor_list[0]
    elif hasattr(spec, '__batch_encoder__'):
        encoded_specs = spec.__batch_encoder__.encoding_specs(spec)
        flat_specs = nest.map_structure(get_batchable_flat_tensor_specs, encoded_specs)
        encoded_flats = nest.pack_sequence_as(flat_specs, tensor_list)
        encoded_value = nest.map_structure_up_to(encoded_specs, batchable_from_tensor_list, encoded_specs, encoded_flats)
        return spec.__batch_encoder__.decode(spec, encoded_value)
    else:
        return spec._from_compatible_tensor_list(tensor_list)