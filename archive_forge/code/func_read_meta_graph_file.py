import copy
from packaging import version as packaging_version  # pylint: disable=g-bad-import-order
import os.path
import re
import sys
from google.protobuf.any_pb2 import Any
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def read_meta_graph_file(filename):
    """Reads a file containing `MetaGraphDef` and returns the protocol buffer.

  Args:
    filename: `meta_graph_def` filename including the path.

  Returns:
    A `MetaGraphDef` protocol buffer.

  Raises:
    IOError: If the file doesn't exist, or cannot be successfully parsed.
  """
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    if not file_io.file_exists(filename):
        raise IOError(f'File does not exist. Received: {filename}.')
    with file_io.FileIO(filename, 'rb') as f:
        file_content = f.read()
    try:
        meta_graph_def.ParseFromString(file_content)
        if sys.byteorder == 'big':
            bst.swap_tensor_content_in_graph_function(meta_graph_def, 'little', 'big')
        return meta_graph_def
    except Exception:
        pass
    try:
        text_format.Merge(file_content.decode('utf-8'), meta_graph_def)
        if sys.byteorder == 'big':
            bst.swap_tensor_content_in_graph_function(meta_graph_def, 'little', 'big')
    except text_format.ParseError as e:
        raise IOError(f'Cannot parse file {filename}: {str(e)}.')
    return meta_graph_def