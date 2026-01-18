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
def get_sparsity_modes(model_object):
    """Get sparsity modes used in a tflite model.

  The sparsity modes are listed in conversion_metadata.fbs file.

  Args:
    model_object: A tflite model in object form.

  Returns:
    The list of sparsity modes used in the model.
  """
    if not model_object or not model_object.metadata:
        return []
    result = set()
    for subgraph in model_object.subgraphs:
        for tensor in subgraph.tensors:
            if not tensor.sparsity:
                continue
            if not tensor.sparsity.blockMap:
                result.add(conversion_metadata_fb.ModelOptimizationMode.RANDOM_SPARSITY)
            else:
                result.add(conversion_metadata_fb.ModelOptimizationMode.BLOCK_SPARSITY)
    return list(result)