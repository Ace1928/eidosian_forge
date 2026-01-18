import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def get_asset_filename_to_add(asset_filepath, asset_filename_map):
    """Get a unique basename to add to the SavedModel if this file is unseen.

  Assets come from users as full paths, and we save them out to the
  SavedModel as basenames. In some cases, the basenames collide. Here,
  we dedupe asset basenames by first checking if the file is the same,
  and, if different, generate and return an index-suffixed basename
  that can be used to add the asset to the SavedModel.

  Args:
    asset_filepath: the full path to the asset that is being saved
    asset_filename_map: a dict of filenames used for saving the asset in
      the SavedModel to full paths from which the filenames were derived.

  Returns:
    Uniquified filename string if the file is not a duplicate, or the original
    filename if the file has already been seen and saved.
  """
    asset_filename = os.path.basename(asset_filepath)
    if asset_filename not in asset_filename_map:
        return asset_filename
    other_asset_filepath = asset_filename_map[asset_filename]
    if other_asset_filepath == asset_filepath:
        return asset_filename
    if not file_io.filecmp(asset_filepath, other_asset_filepath):
        return _get_unique_asset_filename(asset_filename, asset_filename_map)
    return asset_filename