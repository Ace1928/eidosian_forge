import copy
import itertools
import json
import warnings
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers
from keras.src.dtensor import dtensor_api
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import compile_utils
from keras.src.engine import data_adapter
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import steps_per_execution_tuning
from keras.src.engine import training_utils
from keras.src.metrics import base_metric
from keras.src.mixed_precision import loss_scale_optimizer as lso
from keras.src.optimizers import optimizer
from keras.src.optimizers import optimizer_v1
from keras.src.saving import pickle_utils
from keras.src.saving import saving_api
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import model_serialization
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from keras.src.utils import traceback_utils
from keras.src.utils import version_utils
from keras.src.utils.mode_keys import ModeKeys
def get_weight_paths(self):
    """Retrieve all the variables and their paths for the model.

        The variable path (string) is a stable key to identify a `tf.Variable`
        instance owned by the model. It can be used to specify variable-specific
        configurations (e.g. DTensor, quantization) from a global view.

        This method returns a dict with weight object paths as keys
        and the corresponding `tf.Variable` instances as values.

        Note that if the model is a subclassed model and the weights haven't
        been initialized, an empty dict will be returned.

        Returns:
            A dict where keys are variable paths and values are `tf.Variable`
             instances.

        Example:

        ```python
        class SubclassModel(tf.keras.Model):

          def __init__(self, name=None):
            super().__init__(name=name)
            self.d1 = tf.keras.layers.Dense(10)
            self.d2 = tf.keras.layers.Dense(20)

          def call(self, inputs):
            x = self.d1(inputs)
            return self.d2(x)

        model = SubclassModel()
        model(tf.zeros((10, 10)))
        weight_paths = model.get_weight_paths()
        # weight_paths:
        # {
        #    'd1.kernel': model.d1.kernel,
        #    'd1.bias': model.d1.bias,
        #    'd2.kernel': model.d2.kernel,
        #    'd2.bias': model.d2.bias,
        # }

        # Functional model
        inputs = tf.keras.Input((10,), batch_size=10)
        x = tf.keras.layers.Dense(20, name='d1')(inputs)
        output = tf.keras.layers.Dense(30, name='d2')(x)
        model = tf.keras.Model(inputs, output)
        d1 = model.layers[1]
        d2 = model.layers[2]
        weight_paths = model.get_weight_paths()
        # weight_paths:
        # {
        #    'd1.kernel': d1.kernel,
        #    'd1.bias': d1.bias,
        #    'd2.kernel': d2.kernel,
        #    'd2.bias': d2.bias,
        # }
        ```
        """
    result = {}
    descendants, object_paths_dict = tf.__internal__.tracking.ObjectGraphView(self).breadth_first_traversal()
    for descendant in descendants:
        if isinstance(descendant, tf.Variable):
            trackable_references = object_paths_dict[descendant]
            object_path = '.'.join([t.name for t in trackable_references])
            result[object_path] = descendant
    return result