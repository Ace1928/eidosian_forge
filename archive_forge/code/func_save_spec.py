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
def save_spec(self, dynamic_batch=True):
    """Returns the `tf.TensorSpec` of call args as a tuple `(args, kwargs)`.

        This value is automatically defined after calling the model for the
        first time. Afterwards, you can use it when exporting the model for
        serving:

        ```python
        model = tf.keras.Model(...)

        @tf.function
        def serve(*args, **kwargs):
          outputs = model(*args, **kwargs)
          # Apply postprocessing steps, or add additional outputs.
          ...
          return outputs

        # arg_specs is `[tf.TensorSpec(...), ...]`. kwarg_specs, in this
        # example, is an empty dict since functional models do not use keyword
        # arguments.
        arg_specs, kwarg_specs = model.save_spec()

        model.save(path, signatures={
          'serving_default': serve.get_concrete_function(*arg_specs,
                                                         **kwarg_specs)
        })
        ```

        Args:
          dynamic_batch: Whether to set the batch sizes of all the returned
            `tf.TensorSpec` to `None`. (Note that when defining functional or
            Sequential models with `tf.keras.Input([...], batch_size=X)`, the
            batch size will always be preserved). Defaults to `True`.
        Returns:
          If the model inputs are defined, returns a tuple `(args, kwargs)`. All
          elements in `args` and `kwargs` are `tf.TensorSpec`.
          If the model inputs are not defined, returns `None`.
          The model inputs are automatically set when calling the model,
          `model.fit`, `model.evaluate` or `model.predict`.
        """
    return self._get_save_spec(dynamic_batch, inputs_only=False)