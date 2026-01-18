import collections
import copy
import os
from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def compile_args_from_training_config(training_config, custom_objects=None):
    """Return model.compile arguments from training config."""
    if custom_objects is None:
        custom_objects = {}
    with generic_utils.CustomObjectScope(custom_objects):
        optimizer_config = training_config['optimizer_config']
        optimizer = optimizers.deserialize(optimizer_config)
        loss = None
        loss_config = training_config.get('loss', None)
        if loss_config is not None:
            loss = _deserialize_nested_config(losses.deserialize, loss_config)
        metrics = None
        metrics_config = training_config.get('metrics', None)
        if metrics_config is not None:
            metrics = _deserialize_nested_config(_deserialize_metric, metrics_config)
        weighted_metrics = None
        weighted_metrics_config = training_config.get('weighted_metrics', None)
        if weighted_metrics_config is not None:
            weighted_metrics = _deserialize_nested_config(_deserialize_metric, weighted_metrics_config)
        sample_weight_mode = training_config['sample_weight_mode'] if hasattr(training_config, 'sample_weight_mode') else None
        loss_weights = training_config['loss_weights']
    return dict(optimizer=optimizer, loss=loss, metrics=metrics, weighted_metrics=weighted_metrics, loss_weights=loss_weights, sample_weight_mode=sample_weight_mode)