import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def get_zoom_matrix(zooms, image_height, image_width, name=None):
    """Returns projective transform(s) for the given zoom(s).

    Args:
        zooms: A matrix of 2-element lists representing `[zx, zy]`
            to zoom for each image (for a batch of images).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)`. Projective transforms which can be
            given to operation `image_projective_transform_v2`.
            If one row of transforms is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or 'zoom_matrix'):
        num_zooms = tf.shape(zooms)[0]
        x_offset = (image_width - 1.0) / 2.0 * (1.0 - zooms[:, 0, None])
        y_offset = (image_height - 1.0) / 2.0 * (1.0 - zooms[:, 1, None])
        return tf.concat(values=[zooms[:, 0, None], tf.zeros((num_zooms, 1), tf.float32), x_offset, tf.zeros((num_zooms, 1), tf.float32), zooms[:, 1, None], y_offset, tf.zeros((num_zooms, 2), tf.float32)], axis=1)