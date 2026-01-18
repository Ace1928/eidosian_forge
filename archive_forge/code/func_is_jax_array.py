import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils import tree
def is_jax_array(value):
    if hasattr(value, '__class__'):
        for parent in value.__class__.__mro__:
            if parent.__name__ == 'Array' and str(parent.__module__) == 'jax':
                return True
    return is_jax_sparse(value)