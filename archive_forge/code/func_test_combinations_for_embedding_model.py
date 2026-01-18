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
def test_combinations_for_embedding_model():
    eager_mode_strategies = [s for s in strategies_for_embedding_models() if not s.required_tpu]
    return tf.__internal__.test.combinations.times(tf.__internal__.test.combinations.combine(distribution=strategies_for_embedding_models()), graph_mode_test_configuration()) + tf.__internal__.test.combinations.times(tf.__internal__.test.combinations.combine(distribution=eager_mode_strategies), eager_mode_test_configuration())