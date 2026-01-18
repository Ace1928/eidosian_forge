import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from tensorflow.python.util.tf_export import keras_export
def load_own_variables(self, store):
    super().load_own_variables(store)
    self.finalize_state()