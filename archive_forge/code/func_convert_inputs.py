import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def convert_inputs(inputs, dtype=None):
    if isinstance(inputs, dict):
        raise ValueError(f'This layer can only process a tensor representing an image or a batch of images. Received: type(inputs)={type(inputs)}.If you need to pass a dict containing images, labels, and bounding boxes, you should instead use the preprocessing and augmentation layers from `keras_cv.layers`. See docs at https://keras.io/api/keras_cv/layers/')
    inputs = utils.ensure_tensor(inputs, dtype=dtype)
    return inputs