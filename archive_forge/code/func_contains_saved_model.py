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
@tf_export('saved_model.contains_saved_model', v1=[])
def contains_saved_model(export_dir):
    """Checks whether the provided export directory could contain a SavedModel.

  Note that the method does not load any data by itself. If the method returns
  `false`, the export directory definitely does not contain a SavedModel. If the
  method returns `true`, the export directory may contain a SavedModel but
  provides no guarantee that it can be loaded.

  Args:
    export_dir: Absolute path to possible export location. For example,
                '/my/foo/model'.

  Returns:
    True if the export directory contains SavedModel files, False otherwise.
  """
    if isinstance(export_dir, os.PathLike):
        export_dir = os.fspath(export_dir)
    return maybe_saved_model_directory(export_dir)