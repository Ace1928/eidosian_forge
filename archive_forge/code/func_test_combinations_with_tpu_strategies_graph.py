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
def test_combinations_with_tpu_strategies_graph():
    tpu_strategies = [tf.__internal__.distribute.combinations.tpu_strategy]
    return tf.__internal__.test.combinations.times(tf.__internal__.test.combinations.combine(distribution=tpu_strategies), graph_mode_test_configuration())