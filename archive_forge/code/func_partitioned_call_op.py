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
def partitioned_call_op(name: str, args: Sequence[core.Tensor], is_stateful: bool, tout: Sequence[Any], config: Any=None, executor_type: Optional[str]=None, xla_compile_attr: Any=None) -> ops.Operation:
    """Generates a function call op respecting device annotations.

  Args:
    name: Name of the function to call.
    args: The arguments of the function, including captured inputs.
    is_stateful: If the function is stateful.
    tout: a list containing the output dtypes enums
    config: (Optional) A `tensorflow::ConfigProto` proto, serialized. If `None`,
      all optimizations are disabled. Currently only handled for eager defined
      functions.
    executor_type: (Optional) A string for the name of the executor to be used
      in the function call. If not set, or set to an empty string, the default
      tensorflow executor will be used.
    xla_compile_attr: (Optional) value of the XLA compilation attribute.

  Returns:
    Returns the operation.
  """
    if config is None:
        config = function_utils.get_disabled_rewriter_config()
    if executor_type is None:
        executor_type = ''
    args = [ops.convert_to_tensor(x) for x in args]
    tin_attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(type=[x.dtype.as_datatype_enum for x in args]))
    tout_attr = attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(type=tout))
    func_attr = attr_value_pb2.AttrValue(func=attr_value_pb2.NameAttrList(name=name))
    executor_type_attr = attr_value_pb2.AttrValue(s=compat.as_bytes(executor_type))
    config_proto = attr_value_pb2.AttrValue(s=config)
    op_name = 'StatefulPartitionedCall' if is_stateful else 'PartitionedCall'
    op_attrs = {'Tin': tin_attr, 'Tout': tout_attr, 'f': func_attr, 'config_proto': config_proto, 'executor_type': executor_type_attr}
    if xla_compile_attr is not None:
        op_attrs[attributes_lib.XLA_COMPILE] = xla_compile_attr
    op = ops.get_default_graph().create_op(op_name, args, tout, name=op_name, attrs=op_attrs)
    return op