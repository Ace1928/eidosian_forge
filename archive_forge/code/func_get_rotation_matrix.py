import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).

    Args:
        angles: A scalar angle to rotate all images by,
            or (for batches of images) a vector with an angle to
            rotate each image in the batch. The rank must be
            statically known (the shape is not `TensorShape(None)`).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.

    Returns:
        A tensor of shape (num_images, 8).
            Projective transforms which can be given
            to operation `image_projective_transform_v2`.
            If one row of transforms is
            [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or 'rotation_matrix'):
        x_offset = (image_width - 1 - (tf.cos(angles) * (image_width - 1) - tf.sin(angles) * (image_height - 1))) / 2.0
        y_offset = (image_height - 1 - (tf.sin(angles) * (image_width - 1) + tf.cos(angles) * (image_height - 1))) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(values=[tf.cos(angles)[:, None], -tf.sin(angles)[:, None], x_offset[:, None], tf.sin(angles)[:, None], tf.cos(angles)[:, None], y_offset[:, None], tf.zeros((num_angles, 2), tf.float32)], axis=1)