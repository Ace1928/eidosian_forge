import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_image_input_analyser_shape_is_list_of_int():
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32, 3)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert isinstance(analyser.shape, list)
    assert all(map(lambda x: isinstance(x, int), analyser.shape))