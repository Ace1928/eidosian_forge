import copy
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import input_analysers
def test_image_input_with_three_dim():
    analyser = input_analysers.ImageAnalyser()
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(100, 32, 32)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert len(analyser.shape) == 3