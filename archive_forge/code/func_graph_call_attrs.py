import dataclasses
import traceback
import typing
from typing import Any, Dict, List, Optional, Sequence, Union
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_stack
@property
def graph_call_attrs(self) -> Dict[str, Any]:
    """Returns a dictionary of attributes needed to add a call in graph."""
    attrs = {'is_stateful': self.call_options.is_stateful, 'tout': [o.dtype.as_datatype_enum for o in self.function_type.flat_outputs], 'xla_compile_attr': self.cached_definition.attr.get(attributes_lib.XLA_COMPILE, None)}
    attrs.update(self._bound_context.function_call_options.as_attrs())
    return attrs