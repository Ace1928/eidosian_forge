import contextlib
import warnings
import numpy as np
import tensorflow as tf
import tree
from packaging.version import Version
from tensorflow.python.eager import context as tf_context
from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def compiled_loss(self, y, y_pred, sample_weight=None, regularization_losses=None):
    warnings.warn('`model.compiled_loss()` is deprecated. Instead, use `model.compute_loss(x, y, y_pred, sample_weight)`.')
    return self.compute_loss(x=None, y=y, y_pred=y_pred, sample_weight=sample_weight)