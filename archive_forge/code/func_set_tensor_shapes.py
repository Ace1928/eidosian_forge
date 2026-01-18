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
def set_tensor_shapes(tensors, shapes):
    """Sets Tensor shape for each tensor if the shape is defined.

  Args:
    tensors: TensorFlow tensor.Tensor.
    shapes: Dict of strings representing input tensor names to list of
      integers representing input shapes (e.g., {"foo": : [1, 16, 16, 3]}).

  Raises:
    ValueError:
      `shapes` contains an invalid tensor.
      `shapes` contains an invalid shape for a valid tensor.
  """
    if shapes:
        tensor_names_to_tensor = {get_tensor_name(tensor): tensor for tensor in tensors}
        for name, shape in shapes.items():
            if name not in tensor_names_to_tensor:
                raise ValueError("Invalid tensor '{}' found in tensor shapes map.".format(name))
            if shape is not None:
                tensor = tensor_names_to_tensor[name]
                try:
                    tensor.set_shape(shape)
                except ValueError as error:
                    message = "The shape of tensor '{0}' cannot be changed from {1} to {2}. {3}".format(name, tensor.shape, shape, str(error))
                    raise ValueError(message)