import collections
import os
import re
from packaging import version
from tensorflow.compiler.tf2tensorrt import _pywrap_py_utils
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
def get_node_io_shapes(node, key):
    """Returns the input/output shapes of a GraphDef Node."""
    out_shape = []
    for shape in node.attr[key].list.shape:
        out_shape.append([dim.size for dim in shape.dim])
    return out_shape