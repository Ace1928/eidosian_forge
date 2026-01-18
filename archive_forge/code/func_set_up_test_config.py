import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
def set_up_test_config(self, use_numpy=False, use_validation_data=False, with_batch_norm=None):
    self.use_numpy = use_numpy
    self.use_validation_data = use_validation_data
    self.with_batch_norm = with_batch_norm
    keras.backend.set_image_data_format('channels_last')
    np.random.seed(_RANDOM_SEED)
    tf.compat.v1.set_random_seed(_RANDOM_SEED)