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
def get_init_op(meta_graph_def, import_scope=None):
    return _get_op_from_signature_def(meta_graph_def, constants.INIT_OP_SIGNATURE_KEY, import_scope) or _get_op_from_collection(meta_graph_def, constants.MAIN_OP_KEY) or _get_op_from_collection(meta_graph_def, constants.LEGACY_INIT_OP_KEY)