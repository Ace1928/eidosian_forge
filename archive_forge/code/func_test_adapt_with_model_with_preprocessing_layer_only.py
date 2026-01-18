from unittest import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import autokeras as ak
from autokeras import keras_layers
from autokeras import test_utils
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
def test_adapt_with_model_with_preprocessing_layer_only():
    input_node = keras.Input(shape=(10,))
    output_node = keras.layers.experimental.preprocessing.Normalization()(input_node)
    model = keras.Model(input_node, output_node)
    greedy.Greedy.adapt(model, tf.data.Dataset.from_tensor_slices((np.random.rand(100, 10), np.random.rand(100, 10))).batch(32))