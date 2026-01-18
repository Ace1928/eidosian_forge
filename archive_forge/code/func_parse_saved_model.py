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
@tf_export('__internal__.saved_model.parse_saved_model', v1=[])
def parse_saved_model(export_dir):
    """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

  Args:
    export_dir: String or Pathlike, path to the directory containing the
    SavedModel file.

  Returns:
    A `SavedModel` protocol buffer.

  Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
  """
    path_to_pbtxt = file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
    path_to_pb = file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    path_to_cpb = file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_CPB))
    saved_model = saved_model_pb2.SavedModel()
    if file_io.file_exists(path_to_pb):
        with file_io.FileIO(path_to_pb, 'rb') as f:
            file_content = f.read()
        try:
            saved_model.ParseFromString(file_content)
        except message.DecodeError as e:
            raise IOError(f'Cannot parse file {path_to_pb}: {str(e)}.') from e
    elif file_io.file_exists(path_to_pbtxt):
        with file_io.FileIO(path_to_pbtxt, 'rb') as f:
            file_content = f.read()
        try:
            text_format.Parse(file_content.decode('utf-8'), saved_model)
        except text_format.ParseError as e:
            raise IOError(f'Cannot parse file {path_to_pbtxt}: {str(e)}.') from e
    else:
        raise IOError(f'SavedModel file does not exist at: {export_dir}{os.path.sep}{{{constants.SAVED_MODEL_FILENAME_PBTXT}|{constants.SAVED_MODEL_FILENAME_PB}}}')
    return saved_model