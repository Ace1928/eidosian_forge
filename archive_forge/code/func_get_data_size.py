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
def get_data_size(data):
    """Gets the size of data in list, tuple, dict, or a numpy array."""
    assert isinstance(data, (np.ndarray, list, dict, tuple))
    if isinstance(data, np.ndarray):
        return len(data)
    if isinstance(data, (list, tuple)):
        return len(data[0])
    return len(data.values())