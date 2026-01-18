import os as _os
import sys as _sys
from tensorflow.python.util import tf_inspect as _inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['resource_loader.readahead_file_path'])
def readahead_file_path(path, readahead='128M'):
    """Readahead files not implemented; simply returns given path."""
    return path