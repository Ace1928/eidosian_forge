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
def run_graph_optimizations(graph_def, input_arrays, output_arrays, config, graph=None):
    """Apply standard TensorFlow optimizations to the graph_def.

  Args:
    graph_def: Frozen GraphDef to be optimized.
    input_arrays: List of arrays that are considered inputs of the graph.
    output_arrays: List of arrays that are considered outputs of the graph.
    config: tf.ConfigProto.
    graph: TensorFlow Graph. Required when Eager mode is enabled. (default None)

  Returns:
    A new, optimized GraphDef.
  """
    meta_graph = _export_meta_graph(graph_def=graph_def, graph=graph)
    signature = _meta_graph_pb2.SignatureDef()
    for array in input_arrays:
        signature.inputs[array.name].name = array.name
        signature.inputs[array.name].dtype = array.dtype.as_datatype_enum
        signature.inputs[array.name].tensor_shape.CopyFrom(array.shape.as_proto())
    for array in output_arrays:
        signature.outputs[array.name].name = array.name
        signature.outputs[array.name].dtype = array.dtype.as_datatype_enum
        signature.outputs[array.name].tensor_shape.CopyFrom(array.shape.as_proto())
    meta_graph.signature_def['not_used_key'].CopyFrom(signature)
    fetch_collection = _meta_graph_pb2.CollectionDef()
    for array in input_arrays + output_arrays:
        fetch_collection.node_list.value.append(array.name)
    meta_graph.collection_def['train_op'].CopyFrom(fetch_collection)
    return tf_optimizer.OptimizeGraph(config, meta_graph)