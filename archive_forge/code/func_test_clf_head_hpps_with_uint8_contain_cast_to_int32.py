import keras_tuner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
import autokeras as ak
from autokeras import hyper_preprocessors
from autokeras import nodes as input_module
from autokeras import preprocessors
from autokeras import test_utils
from autokeras.blocks import heads as head_module
def test_clf_head_hpps_with_uint8_contain_cast_to_int32():
    dataset = test_utils.generate_one_hot_labels(100, 10, 'dataset')
    dataset = dataset.map(lambda x: tf.cast(x, tf.uint8))
    head = head_module.ClassificationHead(shape=(8,))
    analyser = head.get_analyser()
    for data in dataset:
        analyser.update(data)
    analyser.finalize()
    head.config_from_analyser(analyser)
    assert any([isinstance(hpp, hyper_preprocessors.DefaultHyperPreprocessor) and isinstance(hpp.preprocessor, preprocessors.CastToInt32) for hpp in head.get_hyper_preprocessors()])