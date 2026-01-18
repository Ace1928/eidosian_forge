from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class')
def test_data_io_consistency_input(tuner_fn, tmp_path):
    x1 = test_utils.generate_data()
    y1 = test_utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1,), (y1, y1)))
    dataset_error(dataset, None, dataset, 'Expected x to have', tmp_path)