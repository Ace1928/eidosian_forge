import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import optimizers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import optimizer
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.legacy import optimizer_v2
from keras.src.saving import serialization_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import keras_export
@ema_momentum.setter
def ema_momentum(self, ema_momentum):
    self._optimizer.ema_momentum = ema_momentum