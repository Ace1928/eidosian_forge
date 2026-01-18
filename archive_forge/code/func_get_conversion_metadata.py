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
def get_conversion_metadata(model_buffer):
    """Read conversion metadata from a tflite model.

  Args:
    model_buffer: A tflite model.

  Returns:
    The conversion metadata or None if it is not populated.
  """
    model_object = flatbuffer_utils.convert_bytearray_to_object(model_buffer)
    if not model_object or not model_object.metadata:
        return None
    for meta in model_object.metadata:
        if meta.name.decode('utf-8') == CONVERSION_METADATA_FIELD_NAME:
            metadata_buf = model_object.buffers[meta.buffer].data.tobytes()
            return conversion_metadata_fb.ConversionMetadataT.InitFromObj(conversion_metadata_fb.ConversionMetadata.GetRootAsConversionMetadata(metadata_buf, 0))
    return None