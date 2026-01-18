import collections
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple
from absl import logging
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.eager.polymorphic_function import saved_model_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function as framework_fn
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import versions
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_serialization
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import pywrap_saved_model
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_serialization
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import tracing_utils
from tensorflow.python.saved_model import utils_impl
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import trace_saveable_util
from tensorflow.python.types import core as types_core
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def fill_object_graph_proto(self, proto: saved_object_graph_pb2.SavedObjectGraph):
    """Populate the nodes, children and slot_variables of a SavedObjectGraph."""
    for node_id, node in enumerate(self.nodes):
        assert self.node_ids[node] == node_id
        object_proto = proto.nodes.add()
        object_proto.slot_variables.extend(self._slot_variables.get(node, ()))
        if isinstance(node, _CapturedTensor):
            continue
        for child in self.augmented_graph_view.list_children(node):
            child_proto = object_proto.children.add()
            child_proto.node_id = self.node_ids[child.ref]
            child_proto.local_name = child.name
        for name, ref in self.augmented_graph_view.list_dependencies(node):
            child_proto = object_proto.dependencies.add()
            child_proto.node_id = self.node_ids[ref]
            child_proto.local_name = name
        if node in self._saveable_objects_map:
            assert node not in self._obj_to_registered_saver, "Objects can't have both SaveableObjects and a registered saver"
            for local_name, (save_fn, restore_fn) in self._saveable_objects_map[node].items():
                saveable_object_proto = object_proto.saveable_objects[local_name]
                saveable_object_proto.save_function = self.node_ids[save_fn]
                saveable_object_proto.restore_function = self.node_ids[restore_fn]
        elif node in self._obj_to_registered_saver:
            object_proto.registered_saver = self._obj_to_registered_saver[node]