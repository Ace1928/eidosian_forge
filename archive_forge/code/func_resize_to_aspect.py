import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def resize_to_aspect(x):
    if tf_utils.is_ragged(inputs):
        x = x.to_tensor()
    return image_utils.smart_resize(x, size=size, interpolation=self._interpolation_method)