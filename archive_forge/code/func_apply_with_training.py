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
def apply_with_training(params, state, rng, inputs, training):
    return self.module.apply(self._params_and_state_to_variables(params, state), inputs, rngs=rng, method=self.method, mutable=apply_mutable, training=training)