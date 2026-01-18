from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class')
def test_dataset_and_y(tuner_fn, tmp_path):
    x1 = test_utils.generate_data()
    y1 = test_utils.generate_data(shape=(1,))
    x = tf.data.Dataset.from_tensor_slices((x1, x1))
    y = tf.data.Dataset.from_tensor_slices((y1, y1))
    val_dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(x, y, val_dataset, 'Expected y to be None', tmp_path)