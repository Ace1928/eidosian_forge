import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def random_rotated_inputs(inputs):
    """Rotated inputs with random ops."""
    original_shape = inputs.shape
    unbatched = inputs.shape.rank == 3
    if unbatched:
        inputs = tf.expand_dims(inputs, 0)
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
    img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
    min_angle = self.lower * 2.0 * np.pi
    max_angle = self.upper * 2.0 * np.pi
    angles = self._random_generator.random_uniform(shape=[batch_size], minval=min_angle, maxval=max_angle)
    output = transform(inputs, get_rotation_matrix(angles, img_hd, img_wd), fill_mode=self.fill_mode, fill_value=self.fill_value, interpolation=self.interpolation)
    if unbatched:
        output = tf.squeeze(output, 0)
    output.set_shape(original_shape)
    return output