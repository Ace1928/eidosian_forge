import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def export_meta_graph(self, filename=None, collection_list=None, as_text=False, export_scope=None, clear_devices=False, clear_extraneous_savers=False, strip_default_attrs=False, save_debug_info=False):
    """Writes `MetaGraphDef` to save_path/filename.

    Args:
      filename: Optional meta_graph filename including the path.
      collection_list: List of string keys to collect.
      as_text: If `True`, writes the meta_graph as an ASCII proto.
      export_scope: Optional `string`. Name scope to remove.
      clear_devices: Whether or not to clear the device field for an `Operation`
        or `Tensor` during export.
      clear_extraneous_savers: Remove any Saver-related information from the
        graph (both Save/Restore ops and SaverDefs) that are not associated with
        this Saver.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see [Stripping
        Default-Valued
        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of filename and with `_debug` added before
        the file extension.

    Returns:
      A `MetaGraphDef` proto.
    """
    return export_meta_graph(filename=filename, graph_def=ops.get_default_graph().as_graph_def(add_shapes=True), saver_def=self.saver_def, collection_list=collection_list, as_text=as_text, export_scope=export_scope, clear_devices=clear_devices, clear_extraneous_savers=clear_extraneous_savers, strip_default_attrs=strip_default_attrs, save_debug_info=save_debug_info)