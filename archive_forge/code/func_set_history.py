import inspect
import numpy as np
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
def set_history(self, name, observations):
    if not self.exists(name):
        self.register(name)
    self.metrics[name].set_history(observations)