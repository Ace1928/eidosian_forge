import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def random_flipped_inputs(inputs):
    flipped_outputs = inputs
    if self.horizontal:
        seed = self._random_generator.make_seed_for_stateless_op()
        if seed is not None:
            flipped_outputs = tf.image.stateless_random_flip_left_right(flipped_outputs, seed=seed)
        else:
            flipped_outputs = tf.image.random_flip_left_right(flipped_outputs, self._random_generator.make_legacy_seed())
    if self.vertical:
        seed = self._random_generator.make_seed_for_stateless_op()
        if seed is not None:
            flipped_outputs = tf.image.stateless_random_flip_up_down(flipped_outputs, seed=seed)
        else:
            flipped_outputs = tf.image.random_flip_up_down(flipped_outputs, self._random_generator.make_legacy_seed())
    flipped_outputs.set_shape(inputs.shape)
    return flipped_outputs