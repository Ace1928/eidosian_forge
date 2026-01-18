import numpy as np
import tensorflow as tf
from autokeras import test_utils
from autokeras.preprocessors import common
from autokeras.utils import data_utils
def test_categorical_to_numerical_input_transform():
    x_train = np.array([['a', 'ab', 2.1], ['b', 'bc', 1.0], ['a', 'bc', 'nan']])
    preprocessor = common.CategoricalToNumericalPreprocessor(column_names=['column_a', 'column_b', 'column_c'], column_types={'column_a': 'categorical', 'column_b': 'categorical', 'column_c': 'numerical'})
    dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(32)
    preprocessor.fit(tf.data.Dataset.from_tensor_slices(x_train).batch(32))
    results = preprocessor.transform(dataset)
    for result in results:
        assert result[0][0] == result[2][0]
        assert result[0][0] != result[1][0]
        assert result[0][1] != result[1][1]
        assert result[0][1] != result[2][1]
        assert result[2][2] == 0
        assert result.dtype == tf.float32