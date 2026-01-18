import re
import string
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
def enable_lora(self, rank, a_initializer='he_uniform', b_initializer='zeros'):
    if self.kernel_constraint:
        raise ValueError('Lora is incompatible with kernel constraints. In order to enable lora on this layer, remove the `kernel_constraint` argument.')
    if not self.built:
        raise ValueError("Cannot enable lora on a layer that isn't yet built.")
    if self.lora_enabled:
        raise ValueError('lora is already enabled. This can only be done once per layer.')
    self._tracker.unlock()
    self.lora_kernel_a = self.add_weight(name='lora_kernel_a', shape=self.kernel.shape[:-1] + (rank,), initializer=initializers.get(a_initializer), regularizer=self.kernel_regularizer)
    self.lora_kernel_b = self.add_weight(name='lora_kernel_b', shape=(rank, self.kernel.shape[-1]), initializer=initializers.get(b_initializer), regularizer=self.kernel_regularizer)
    self._kernel.trainable = False
    self._tracker.lock()
    self.lora_enabled = True
    self.lora_rank = rank