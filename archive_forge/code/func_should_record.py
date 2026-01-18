import logging
import os
import sys
import time
import warnings
import tree
from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.layers import Embedding
from keras.src.optimizers import Optimizer
from keras.src.utils import file_utils
def should_record():
    return step % self.update_freq == 0