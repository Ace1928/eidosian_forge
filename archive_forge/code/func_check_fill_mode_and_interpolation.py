import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {'reflect', 'wrap', 'constant', 'nearest'}:
        raise NotImplementedError(f'Unknown `fill_mode` {fill_mode}. Only `reflect`, `wrap`, `constant` and `nearest` are supported.')
    if interpolation not in {'nearest', 'bilinear'}:
        raise NotImplementedError(f'Unknown `interpolation` {interpolation}. Only `nearest` and `bilinear` are supported.')