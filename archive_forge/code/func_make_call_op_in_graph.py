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
def make_call_op_in_graph(atomic: AtomicFunction, tensor_inputs: Sequence[core.Tensor], context_call_attrs: Dict[str, Any]):
    """Adds an AtomicFunction to graph."""
    graph = ops.get_default_graph()
    graph._add_function_recursive(atomic)
    op = partitioned_call_op(name=atomic.name, args=tensor_inputs, is_stateful=atomic.call_options.is_stateful, tout=[o.dtype.as_datatype_enum for o in atomic.function_type.flat_outputs], config=context_call_attrs['config_proto'], executor_type=context_call_attrs['executor_type'], xla_compile_attr=atomic.cached_definition.attr.get(attributes_lib.XLA_COMPILE, None))
    _set_read_only_resource_inputs_attr(op, atomic.graph)
    ops.set_int_list_attr(op, acd.COLLECTIVE_MANAGER_IDS, atomic._call_options.collective_manager_ids_used)
    return op.outputs