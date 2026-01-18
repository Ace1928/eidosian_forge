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
def get_best_model(self):
    with keras_tuner.engine.tuner.maybe_distribute(self.distribution_strategy):
        model = keras.models.load_model(self.best_model_path)
    return model