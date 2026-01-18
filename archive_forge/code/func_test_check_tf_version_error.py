import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_check_tf_version_error():
    utils.tf.__version__ = '2.1.0'
    with pytest.warns(ImportWarning) as record:
        utils.check_tf_version()
    assert len(record) == 1
    assert 'Tensorflow package version needs to be at least' in record[0].message.args[0]