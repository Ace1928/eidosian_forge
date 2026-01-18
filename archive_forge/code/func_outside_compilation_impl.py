from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def outside_compilation_impl(is_map, computation: Callable[..., Any], *args, **kwargs) -> Any:
    """Tags ops in `computation` with outside compilation attributes for ordinary `outside_compilation` or `map_outside_compilation`."""
    args = [] if args is None else args
    graph = ops.get_default_graph()
    if isinstance(graph, func_graph.FuncGraph):
        try:
            tpu_context, _ = _enclosing_tpu_context_and_graph()
        except ValueError:
            logging.warning('Outside compilation attempted outside TPUReplicateContext scope. As no enclosing TPUReplicateContext can be found, returning the result of `computation` as is.')
            return computation(*args, **kwargs)
        outside_compilation_name = str(tpu_context._outside_compilation_counter)
        tpu_context._outside_compilation_counter = tpu_context._outside_compilation_counter + 1
        outside_compilation_context = OutsideCompilationV2Context(outside_compilation_name, is_map_outside_compilation=is_map)
        outside_compilation_context.Enter()
        args = [] if args is None else args
        retval = computation(*args, **kwargs)
        outside_compilation_context.Exit()
        return retval
    initial_context = graph._get_control_flow_context()
    context = initial_context
    while context:
        if isinstance(context, TPUReplicateContext):
            context._EnterOutsideCompilationScope(is_map_outside_compilation=is_map)
        context = context.outer_context
    retval = computation(*args, **kwargs)
    final_context = graph._get_control_flow_context()
    if initial_context is not final_context:
        raise NotImplementedError('Control-flow context cannot be different at start and end of an outside_compilation scope')
    context = initial_context
    while context:
        if isinstance(context, TPUReplicateContext):
            context._ExitOutsideCompilationScope()
        context = context.outer_context
    return retval