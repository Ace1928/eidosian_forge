import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def random_contrasted_inputs(inputs):
    seed = self._random_generator.make_seed_for_stateless_op()
    if seed is not None:
        output = tf.image.stateless_random_contrast(inputs, 1.0 - self.lower, 1.0 + self.upper, seed=seed)
    else:
        output = tf.image.random_contrast(inputs, 1.0 - self.lower, 1.0 + self.upper, seed=self._random_generator.make_legacy_seed())
    output = tf.clip_by_value(output, 0, 255)
    output.set_shape(inputs.shape)
    return output