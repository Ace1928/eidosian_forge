import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_infer_single_column_two_classes():
    analyser = output_analysers.ClassificationAnalyser(name='a')
    dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0, 2, 10)).batch(32)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.num_classes == 2