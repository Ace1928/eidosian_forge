import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_get_hyperparameter_with_none_return_hp():
    hp = utils.get_hyperparameter(None, hyperparameters.Choice('hp', [10, 20]), int)
    assert isinstance(hp, hyperparameters.Choice)