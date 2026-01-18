from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class')
def test_export_model(tuner_fn, tmp_path):
    tuner_class = tuner_fn.return_value
    tuner = tuner_class.return_value
    x_train = np.random.rand(100, 32, 32, 3)
    y_train = np.random.rand(100, 1)
    auto_model = get_single_io_auto_model(tmp_path)
    auto_model.fit(x_train, y_train, epochs=2, validation_data=(x_train, y_train))
    auto_model.export_model()
    assert tuner.get_best_model.called