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
def object_graph_key_mapping(checkpoint_path):
    """Return name to key mappings from the checkpoint.

  Args:
    checkpoint_path: string, path to object-based checkpoint

  Returns:
    Dictionary mapping tensor names to checkpoint keys.
  """
    reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
    object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
    object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
    object_graph_proto.ParseFromString(object_graph_string)
    names_to_keys = {}
    for node in object_graph_proto.nodes:
        for attribute in node.attributes:
            names_to_keys[attribute.full_name] = attribute.checkpoint_key
    return names_to_keys