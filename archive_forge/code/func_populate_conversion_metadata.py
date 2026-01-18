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
def populate_conversion_metadata(model_object, metadata):
    """Add or update conversion metadata to a tflite model.

  Args:
    model_object: A tflite model in object form.
    metadata: The conversion metadata.

  Returns:
    A tflite model object with embedded conversion metadata.
  """
    try:
        metadata_builder = flatbuffers.Builder(0)
        metadata_builder.Finish(metadata.Pack(metadata_builder))
        buffer_field = schema_fb.BufferT()
        buffer_field.data = metadata_builder.Output()
        if not model_object.metadata:
            model_object.metadata = []
        else:
            for meta in model_object.metadata:
                if meta.name.decode('utf-8') == CONVERSION_METADATA_FIELD_NAME:
                    model_object.buffers[meta.buffer] = buffer_field
                    return model_object
        if not model_object.buffers:
            model_object.buffers = []
        model_object.buffers.append(buffer_field)
        metadata_field = schema_fb.MetadataT()
        metadata_field.name = CONVERSION_METADATA_FIELD_NAME
        metadata_field.buffer = len(model_object.buffers) - 1
        model_object.metadata.append(metadata_field)
        return model_object
    except Exception:
        return model_object