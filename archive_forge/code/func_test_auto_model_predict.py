from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class', side_effect=get_tuner_class)
def test_auto_model_predict(tuner_fn, tmp_path):
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)
    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_split=0.2)
    auto_model.predict(x_train)
    assert tuner_fn.called