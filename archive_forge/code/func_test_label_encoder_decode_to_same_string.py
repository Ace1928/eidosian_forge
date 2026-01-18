import numpy as np
import tensorflow as tf
from autokeras import preprocessors
from autokeras.preprocessors import encoders
from autokeras.utils import data_utils
def test_label_encoder_decode_to_same_string():
    encoder = encoders.LabelEncoder(['a', 'b'])
    result = encoder.postprocess([[0], [1]])
    assert np.array_equal(result, np.array([['a'], ['b']]))