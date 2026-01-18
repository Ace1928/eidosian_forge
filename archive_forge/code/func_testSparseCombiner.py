from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator.canned import linear
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.export import export
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.inputs import pandas_io
def testSparseCombiner(self):
    w_a = 2.0
    w_b = 3.0
    w_c = 5.0
    bias = 5.0
    with tf.Graph().as_default():
        tf.Variable([[w_a], [w_b], [w_c]], name=LANGUAGE_WEIGHT_NAME)
        tf.Variable([bias], name=BIAS_NAME)
        tf.Variable(1, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)

    def _input_fn():
        return tf.compat.v1.data.Dataset.from_tensors({'language': tf.sparse.SparseTensor(values=['a', 'c', 'b', 'c'], indices=[[0, 0], [0, 1], [1, 0], [1, 1]], dense_shape=[2, 2])})
    feature_columns = (self._fc_lib.categorical_column_with_vocabulary_list('language', vocabulary_list=['a', 'b', 'c']),)
    linear_classifier = self._linear_classifier_fn(feature_columns=feature_columns, model_dir=self._model_dir)
    predictions = linear_classifier.predict(input_fn=_input_fn)
    predicted_scores = list([x['logits'] for x in predictions])
    self.assertAllClose([[12.0], [13.0]], predicted_scores)
    linear_classifier = self._linear_classifier_fn(feature_columns=feature_columns, model_dir=self._model_dir, sparse_combiner='mean')
    predictions = linear_classifier.predict(input_fn=_input_fn)
    predicted_scores = list([x['logits'] for x in predictions])
    self.assertAllClose([[8.5], [9.0]], predicted_scores)
    linear_classifier = self._linear_classifier_fn(feature_columns=feature_columns, model_dir=self._model_dir, sparse_combiner='sqrtn')
    predictions = linear_classifier.predict(input_fn=_input_fn)
    predicted_scores = list([x['logits'] for x in predictions])
    self.assertAllClose([[9.94974], [10.65685]], predicted_scores)