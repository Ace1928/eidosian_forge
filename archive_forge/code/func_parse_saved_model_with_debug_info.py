import os
import sys
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def parse_saved_model_with_debug_info(export_dir):
    """Reads the savedmodel as well as the graph debug info.

  Args:
    export_dir: Directory containing the SavedModel and GraphDebugInfo files.

  Returns:
    `SavedModel` and `GraphDebugInfo` protocol buffers.

  Raises:
    IOError: If the saved model file does not exist, or cannot be successfully
    parsed. Missing graph debug info file is fine.
  """
    saved_model = parse_saved_model(export_dir)
    debug_info_path = file_io.join(path_helpers.get_debug_dir(export_dir), constants.DEBUG_INFO_FILENAME_PB)
    debug_info = graph_debug_info_pb2.GraphDebugInfo()
    if file_io.file_exists(debug_info_path):
        with file_io.FileIO(debug_info_path, 'rb') as debug_file:
            try:
                debug_info.ParseFromString(debug_file.read())
            except message.DecodeError as e:
                raise IOError(f'Cannot parse file {debug_info_path}: {e}.')
    return (saved_model, debug_info)