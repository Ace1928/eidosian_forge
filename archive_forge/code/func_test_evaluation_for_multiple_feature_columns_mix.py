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
def test_evaluation_for_multiple_feature_columns_mix(self):
    with tf.Graph().as_default():
        tf.Variable([[10.0]], name=AGE_WEIGHT_NAME)
        tf.Variable([[2.0]], name=HEIGHT_WEIGHT_NAME)
        tf.Variable([5.0], name=BIAS_NAME)
        tf.Variable(100, name=tf.compat.v1.GraphKeys.GLOBAL_STEP, dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    batch_size = 2
    feature_columns = [feature_column.numeric_column('age'), tf.feature_column.numeric_column('height')]

    def _input_fn():
        features_ds = tf.compat.v1.data.Dataset.from_tensor_slices({'age': np.array([20, 40]), 'height': np.array([4, 8])})
        labels_ds = tf.compat.v1.data.Dataset.from_tensor_slices(np.array([[213.0], [421.0]]))
        return tf.compat.v1.data.Dataset.zip((features_ds, labels_ds)).batch(batch_size).repeat(None)
    est = self._linear_regressor_fn(feature_columns=feature_columns, model_dir=self._model_dir)
    eval_metrics = est.evaluate(input_fn=_input_fn, steps=1)
    self.assertItemsEqual((metric_keys.MetricKeys.LOSS, metric_keys.MetricKeys.LOSS_MEAN, metric_keys.MetricKeys.PREDICTION_MEAN, metric_keys.MetricKeys.LABEL_MEAN, tf.compat.v1.GraphKeys.GLOBAL_STEP), eval_metrics.keys())
    self.assertAlmostEqual(0, eval_metrics[metric_keys.MetricKeys.LOSS])