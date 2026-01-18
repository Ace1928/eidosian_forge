import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
def register_uninitialized_variable(variable):
    uninitialized_variables = global_state.get_global_attribute('uninitialized_variables', [], set_to_default=True)
    uninitialized_variables.append(variable)