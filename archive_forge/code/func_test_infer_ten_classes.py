import numpy as np
import pytest
import tensorflow as tf
from autokeras import test_utils
from autokeras.analysers import output_analysers
def test_infer_ten_classes():
    analyser = output_analysers.ClassificationAnalyser(name='a')
    dataset = test_utils.generate_one_hot_labels(dtype='dataset', num_classes=10)
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    assert analyser.num_classes == 10