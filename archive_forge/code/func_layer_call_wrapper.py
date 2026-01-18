import functools
import threading
import weakref
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def layer_call_wrapper(call_collection, method, name):
    """Ensures layer losses are kept the same, and runs method in call context."""

    def wrapper(*args, **kwargs):
        """Calls method within call context."""
        layer = call_collection.layer
        training = None
        inputs = _filtered_inputs([args, kwargs])
        if (args or kwargs) and call_collection.training_arg_was_passed(args, kwargs):
            training = call_collection.get_training_arg_value(args, kwargs)
        original_losses = _reset_layer_losses(layer)
        with base_layer_utils.call_context().enter(layer, inputs=inputs, build_graph=False, training=training, saving=True):
            with autocast_variable.enable_auto_cast_variables(layer._compute_dtype_object):
                ret = method(*args, **kwargs)
        _restore_layer_losses(original_losses)
        return ret
    fn = tf_decorator.make_decorator(target=method, decorator_func=wrapper)
    fn.__name__ = name
    return fn