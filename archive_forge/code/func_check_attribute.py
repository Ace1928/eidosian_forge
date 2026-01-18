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
def check_attribute(attribute_self, attribute_other):
    nonlocal is_subtype
    if not is_subtype:
        return
    if isinstance(attribute_self, trace.TraceType):
        if not attribute_self.is_subtype_of(attribute_other):
            is_subtype = False
            return
    elif attribute_self != attribute_other:
        is_subtype = False