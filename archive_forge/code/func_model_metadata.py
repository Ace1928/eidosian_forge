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
def model_metadata(model, include_optimizer=True, require_config=True):
    """Returns a dictionary containing the model metadata."""
    from tensorflow.python.keras import __version__ as keras_version
    from tensorflow.python.keras.optimizer_v2 import optimizer_v2
    model_config = {'class_name': model.__class__.__name__}
    try:
        model_config['config'] = model.get_config()
    except NotImplementedError as e:
        if require_config:
            raise e
    metadata = dict(keras_version=str(keras_version), backend=K.backend(), model_config=model_config)
    if model.optimizer and include_optimizer:
        if isinstance(model.optimizer, optimizer_v1.TFOptimizer):
            logging.warning('TensorFlow optimizers do not make it possible to access optimizer attributes or optimizer state after instantiation. As a result, we cannot save the optimizer as part of the model save file. You will have to compile your model again after loading it. Prefer using a Keras optimizer instead (see keras.io/optimizers).')
        elif model._compile_was_called:
            training_config = model._get_compile_args(user_metrics=False)
            training_config.pop('optimizer', None)
            metadata['training_config'] = _serialize_nested_config(training_config)
            if isinstance(model.optimizer, optimizer_v2.RestoredOptimizer):
                raise NotImplementedError("As of now, Optimizers loaded from SavedModel cannot be saved. If you're calling `model.save` or `tf.keras.models.save_model`, please set the `include_optimizer` option to `False`. For `tf.saved_model.save`, delete the optimizer from the model.")
            else:
                optimizer_config = {'class_name': generic_utils.get_registered_name(model.optimizer.__class__), 'config': model.optimizer.get_config()}
            metadata['training_config']['optimizer_config'] = optimizer_config
    return metadata