import traceback
from typing import Any, Callable, Hashable
import weakref
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.saved_model import save_context
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@tf_contextlib.contextmanager
def inner_cm():
    """Context manager for copying distribute.Strategy scope information."""
    graph = ops.get_default_graph()
    old_strategy_stack = self._distribution_strategy_stack
    self._distribution_strategy_stack = list(graph._distribution_strategy_stack)
    old_device_stack = self._device_function_stack
    if not context.executing_eagerly() and (device_stack_has_callable(graph._device_function_stack) or (self._distribution_strategy_stack and (not ops.executing_eagerly_outside_functions()))):
        self._device_function_stack = graph._device_function_stack.copy()
    old_creator_stack = self._variable_creator_stack
    self._variable_creator_stack = graph._variable_creator_stack
    old_graph_key = self._graph_key
    self._graph_key = graph._graph_key
    old_scope_exit_callbacks = self._scope_exit_callbacks
    self._scope_exit_callbacks = []
    with outer_cm as g:
        try:
            yield g
        finally:
            try:
                for fn in self._scope_exit_callbacks:
                    fn()
            finally:
                self._scope_exit_callbacks = old_scope_exit_callbacks
                self._distribution_strategy_stack = old_strategy_stack
                self._device_function_stack = old_device_stack
                self._variable_creator_stack = old_creator_stack
                self._graph_key = old_graph_key