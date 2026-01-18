import copy
import itertools
import threading
import types
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer_utils
from keras.src.utils import control_flow_util
from keras.src.utils import tf_contextlib
from keras.src.utils.generic_utils import LazyLoader
from keras.src.utils.layer_utils import CallFunctionSpec
def set_training_arg_spec(arg_spec, default_training_value):
    """Set `training=DEFAULT` argument in an ArgSpec."""
    if 'training' in arg_spec.args:
        index = arg_spec.args.index('training')
        training_default_index = len(arg_spec.args) - index
        defaults = list(arg_spec.defaults) if arg_spec.defaults is not None else []
        if arg_spec.defaults and len(arg_spec.defaults) >= training_default_index and (defaults[-training_default_index] is None):
            defaults[-training_default_index] = default_training_value
            return arg_spec._replace(defaults=defaults)
    elif 'training' not in arg_spec.kwonlyargs:
        kwonlyargs = arg_spec.kwonlyargs + ['training']
        kwonlydefaults = copy.copy(arg_spec.kwonlydefaults) or {}
        kwonlydefaults['training'] = default_training_value
        return arg_spec._replace(kwonlyargs=kwonlyargs, kwonlydefaults=kwonlydefaults)
    return arg_spec