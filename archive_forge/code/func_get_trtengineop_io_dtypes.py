import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def get_trtengineop_io_dtypes(node, key):
    """Returns the input/output dtypes of a TRTEngineOp."""
    return _convert_dtype_id_to_str(node.attr[key].list.type)