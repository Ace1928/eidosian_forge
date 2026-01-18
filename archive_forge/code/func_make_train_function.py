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
def make_train_function(self):
    """Creates a function that executes one step of training.

    This method can be overridden to support custom training logic.
    This method is called by `Model.fit` and `Model.train_on_batch`.

    Typically, this method directly controls `tf.function` and
    `tf.distribute.Strategy` settings, and delegates the actual training
    logic to `Model.train_step`.

    This function is cached the first time `Model.fit` or
    `Model.train_on_batch` is called. The cache is cleared whenever
    `Model.compile` is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, and return a `dict` containing values that will
      be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
      `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self.train_function is not None:
        return self.train_function

    def step_function(model, iterator):
        """Runs a single training step."""

        def run_step(data):
            outputs = model.train_step(data)
            with ops.control_dependencies(_minimum_control_deps(outputs)):
                model._train_counter.assign_add(1)
            return outputs
        data = next(iterator)
        outputs = model.distribute_strategy.run(run_step, args=(data,))
        outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction='first')
        write_scalar_summaries(outputs, step=model._train_counter)
        return outputs
    if self._steps_per_execution.numpy().item() == 1:

        def train_function(iterator):
            """Runs a training execution with one step."""
            return step_function(self, iterator)
    else:

        def train_function(iterator):
            """Runs a training execution with multiple steps."""
            for _ in math_ops.range(self._steps_per_execution):
                outputs = step_function(self, iterator)
            return outputs
    if not self.run_eagerly:
        train_function = def_function.function(train_function, experimental_relax_shapes=True)
        self.train_tf_function = train_function
    self.train_function = train_function
    if self._cluster_coordinator:
        self.train_function = lambda iterator: self._cluster_coordinator.schedule(train_function, args=(iterator,))
    return self.train_function