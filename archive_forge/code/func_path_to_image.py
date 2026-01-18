import json
import multiprocessing
import os
from typing import Optional
from typing import Tuple
import numpy as np
import tensorflow as tf
from keras_tuner.engine import hyperparameters
def path_to_image(image, num_channels, image_size, interpolation):
    image = tf.io.read_file(image)
    image = tf.io.decode_image(image, channels=num_channels, expand_animations=False)
    image = tf.image.resize(image, image_size, method=interpolation)
    image.set_shape((image_size[0], image_size[1], num_channels))
    return image