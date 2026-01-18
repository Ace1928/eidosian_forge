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
def modify_model_io_type(model, inference_input_type=dtypes.float32, inference_output_type=dtypes.float32):
    """Modify the input/output type of a tflite model.

  Args:
    model: A tflite model.
    inference_input_type: tf.DType representing modified input type.
      (default tf.float32. If model input is int8 quantized, it must be in
      {tf.float32, tf.int8,tf.uint8}, else if model input is int16 quantized,
      it must be in {tf.float32, tf.int16}, else it must be tf.float32)
    inference_output_type: tf.DType representing modified output type.
      (default tf.float32. If model output is int8 dequantized, it must be in
      {tf.float32, tf.int8,tf.uint8}, else if model output is int16 dequantized,
      it must be in {tf.float32, tf.int16}, else it must be tf.float32)
  Returns:
    A tflite model with modified input/output type.

  Raises:
    ValueError: If `inference_input_type`/`inference_output_type` is unsupported
      or a supported integer type is specified for a model whose input/output is
      not quantized/dequantized.
    RuntimeError: If the modification was unsuccessful.

  """
    if inference_input_type == dtypes.float32 and inference_output_type == dtypes.float32:
        return model
    model_object = _convert_model_from_bytearray_to_object(model)
    _modify_model_input_type(model_object, inference_input_type)
    _modify_model_output_type(model_object, inference_output_type)
    _remove_redundant_quantize_ops(model_object)
    return _convert_model_from_object_to_bytearray(model_object)