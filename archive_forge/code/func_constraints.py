import copy
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.base_wrapper import Wrapper
from keras.src.saving import serialization_lib
from keras.src.utils import generic_utils
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
@property
def constraints(self):
    constraints = {}
    if hasattr(self.forward_layer, 'constraints'):
        constraints.update(self.forward_layer.constraints)
        constraints.update(self.backward_layer.constraints)
    return constraints