import copy
import itertools
import json
import os
import warnings
import weakref
from tensorflow.python.autograph.lang import directives
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer as lso
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import constants as sm_constants
from tensorflow.python.saved_model import loader_impl as sm_loader
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.tools.docs import doc_controls
def predict_function(iterator):
    """Runs an evaluation execution with multiple steps."""
    outputs = step_function(self, iterator)
    for _ in math_ops.range(self._steps_per_execution - 1):
        directives.set_loop_options(shape_invariants=[(t, tf_utils.get_tensor_spec(t, dynamic_batch=True).shape) for t in nest.flatten(outputs)])
        step_outputs = step_function(self, iterator)
        outputs = nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs, step_outputs)
    return outputs