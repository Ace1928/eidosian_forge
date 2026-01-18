import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def get_bin_boundaries(summary, num_bins):
    return compress(summary, 1.0 / num_bins)[0, :-1]