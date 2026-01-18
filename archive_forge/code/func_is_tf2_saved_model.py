import collections
import functools
import os
import sys
from absl import logging
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import restore
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.polymorphic_function import saved_model_utils as function_saved_model_utils
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def is_tf2_saved_model(export_dir):
    """Identifies if an exported SavedModel is a TF2 SavedModel.

  There are differences in SavedModel semantics between TF1 and TF2 that are
  documented here:
  https://www.tensorflow.org/guide/migrate/saved_model#savedmodel. This helper
  util function serves to distinguish the TF1 vs TF2 semantics used when
  exporting SavedModels.

  Args:
    export_dir: The SavedModel directory to load from.

  Returns:
    True if TF2 SavedModel semantics are used, False if TF1 SavedModel semantics
    are used.
  """
    try:
        fingerprint = fingerprinting.read_fingerprint(export_dir)
        if fingerprint.saved_object_graph_hash != 0:
            logging.info('SavedModel at %s is a TF2 SavedModel', export_dir)
            return True
    except Exception:
        logging.info('Failed to read fingerprint from SavedModel. Parsing MetaGraph ...')
        saved_model_proto = loader_impl.parse_saved_model(export_dir)
        if len(saved_model_proto.meta_graphs) == 1 and saved_model_proto.meta_graphs[0].HasField('object_graph_def'):
            logging.info('SavedModel at %s is a TF2 SavedModel', export_dir)
            return True
    logging.info('SavedModel at %s is a TF1 SavedModel', export_dir)
    return False