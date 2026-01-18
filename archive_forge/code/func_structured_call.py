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
def structured_call(self, args: Sequence[Any], kwargs: Dict[str, Any], captures: Sequence[Any]) -> Any:
    """Calls with structured tensor inputs and returns structured output."""
    bound_parameters = self.function_type.bind(*args, **kwargs)
    tensor_inputs = self.function_type.unpack_inputs(bound_parameters)
    capture_inputs = self.function_type.unpack_captures(captures)
    return self.flat_call(tensor_inputs + capture_inputs)