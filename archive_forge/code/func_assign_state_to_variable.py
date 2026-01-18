import inspect
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import jax_utils
from keras.src.utils import tracking
from keras.src.utils import tree
from keras.src.utils.module_utils import jax
def assign_state_to_variable(value, variable):
    if not hasattr(variable, 'assign'):
        raise ValueError('Structure mismatch: the structure of the state returned by `call` does not match the structure of the state at initialization time.')
    variable.assign(value)