import random as python_random
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils import jax_utils
def seed_initializer(*args, **kwargs):
    dtype = kwargs.get('dtype', None)
    return self.backend.convert_to_tensor([seed, 0], dtype=dtype)