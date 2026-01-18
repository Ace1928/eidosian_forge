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
def get_input_for_dynamic_lr_test(self, **kwargs):
    """Generates inputs that are dictionaries.

        We only provide a default implementation of this method here. If you
        need more customized way of providing input to your model, overwrite
        this method.

        Args:
          **kwargs: key word arguments about how to create the input
            dictionaries

        Returns:
          Three dictionaries representing the input for fit(), evaluate() and
          predict()
        """
    training_input = kwargs
    return (training_input, None, None)