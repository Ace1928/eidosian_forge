from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
@mock.patch('autokeras.auto_model.get_tuner_class')
def test_multi_io_with_tf_dataset_doesnt_crash(tuner_fn, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    x1 = test_utils.generate_data()
    y1 = test_utils.generate_data(shape=(1,))
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x1), (y1, y1)))
    auto_model.fit(dataset, epochs=2)