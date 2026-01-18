import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@property
def node_names(self):
    """Lazily creates a mapping from node id to ("path", "to", "root")."""
    if self._node_name_cache is not None:
        return self._node_name_cache
    path_to_root = {}
    path_to_root[0] = ('(root)',)
    to_visit = collections.deque([0])
    while to_visit:
        node_id = to_visit.popleft()
        obj = self._object_graph_proto.nodes[node_id]
        for child in obj.children:
            if child.node_id not in path_to_root:
                path_to_root[child.node_id] = path_to_root[node_id] + (child.local_name,)
                to_visit.append(child.node_id)
    node_names = {}
    for node_id, path_to_root in path_to_root.items():
        node_names[node_id] = '.'.join(path_to_root)
    for node_id, node in enumerate(self._object_graph_proto.nodes):
        for slot_reference in node.slot_variables:
            node_names[slot_reference.slot_variable_node_id] = f"{node_names[node_id]}'s state '{slot_reference.slot_name}' for {node_names[slot_reference.original_variable_node_id]}"
    self._node_name_cache = node_names
    return node_names