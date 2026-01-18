import dataclasses
import functools
import os
import threading
import types as types_lib
import weakref
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export
def class_method_to_instance_method(original_function, instance):
    """Constructs a new `Function` with `self` bound."""
    weak_instance = weakref.ref(instance)
    bound_method = types_lib.MethodType(original_function.python_function, tf_method_target.TfMethodTarget(weak_instance, original_function.python_function))
    assert hasattr(original_function, '_name')
    assert hasattr(original_function, '_autograph')
    assert hasattr(original_function, '_function_type')
    assert hasattr(original_function, 'python_function')
    weak_bound_method_wrapper = None

    def bound_method_wrapper(*args, **kwargs):
        """Wraps either a dummy MethodType or a converted AutoGraph function."""
        strong_bound_method_wrapper = weak_bound_method_wrapper()
        wrapped_fn = strong_bound_method_wrapper.__wrapped__
        if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
            wrapped_fn = original_function.python_function
            return wrapped_fn(weak_instance(), *args, **kwargs)
        return wrapped_fn(*args, **kwargs)
    weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)
    instance_func = type(original_function)(tf_decorator.make_decorator(bound_method, bound_method_wrapper), name=original_function._name, autograph=original_function._autograph, input_signature=original_function.input_signature, reduce_retracing=original_function._reduce_retracing, jit_compile=original_function._jit_compile)
    wrapped_instance_func = tf_decorator.make_decorator(bound_method, instance_func)
    return wrapped_instance_func