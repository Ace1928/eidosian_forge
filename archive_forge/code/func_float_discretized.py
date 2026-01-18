import tree
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import saving_lib
from keras.src.saving import serialization_lib
from keras.src.utils import backend_utils
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
@classmethod
def float_discretized(cls, num_bins, bin_boundaries=None, output_mode='one_hot', name=None):
    name = name or auto_name('float_discretized')
    preprocessor = layers.Discretization(num_bins=num_bins, bin_boundaries=bin_boundaries, name=f'{name}_preprocessor')
    return Feature(dtype='float32', preprocessor=preprocessor, output_mode=output_mode)