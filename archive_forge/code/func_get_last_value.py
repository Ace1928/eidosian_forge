import inspect
import numpy as np
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
def get_last_value(self, name):
    self._assert_exists(name)
    return self.metrics[name].get_last_value()