import copy
import datetime
import sys
from absl import logging
import flatbuffers
from tensorflow.core.protobuf import config_pb2 as _config_pb2
from tensorflow.core.protobuf import meta_graph_pb2 as _meta_graph_pb2
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.lite.python import tflite_keras_util as _tflite_keras_util
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import find_all_hinted_output_nodes
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.eager import function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation as _error_interpolation
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph as _export_meta_graph
def build_debug_info_func(original_graph):
    """Returns a method to retrieve the `GraphDebugInfo` from the original graph.

  Args:
    original_graph: The original `Graph` containing all the op stack traces.

  Returns:
    A function which retrieves the stack traces from the original graph and
    converts them to a `GraphDebugInfo` for a given set of nodes.
  """

    def f(original_nodes):
        """Function to create `GraphDebugInfo` for the given `original_nodes`."""
        if not original_graph:
            return None
        useful_ops = []
        for func, name in original_nodes:
            try:
                if not func:
                    useful_ops.append((func, original_graph.get_operation_by_name(name)))
                else:
                    sub_func = original_graph._get_function(func)
                    if isinstance(sub_func, function.AtomicFunction):
                        useful_ops.append((func, sub_func.graph.get_operation_by_name(name)))
                    else:
                        sys.stderr.write("Use '@tf.function' or '@defun' to decorate the function.\n")
                        continue
            except KeyError:
                continue
        return _error_interpolation.create_graph_debug_info_def(useful_ops)
    return f