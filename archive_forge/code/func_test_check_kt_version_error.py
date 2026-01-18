import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_check_kt_version_error():
    utils.keras_tuner.__version__ = '1.0.0'
    with pytest.warns(ImportWarning) as record:
        utils.check_kt_version()
    assert len(record) == 1
    assert 'Keras Tuner package version needs to be at least' in record[0].message.args[0]