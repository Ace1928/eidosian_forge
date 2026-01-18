import collections
import copy
import os
import keras_tuner
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils
def get_output_layers(tensor):
    output_layers = []
    tensor = nest.flatten(tensor)[0]
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        input_node = nest.flatten(layer.input)[0]
        if input_node is tensor:
            if isinstance(layer, preprocessing.PreprocessingLayer):
                output_layers.append(layer)
    return output_layers