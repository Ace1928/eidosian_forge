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
def testTwoFeatureColumns(self):
    """Tests predict with two feature columns."""
    with tf.Graph().as_default():
        tf.Variable([[10.0]], name='linear/linear_model/x0/weights')
        tf.Variable([[20.0]], name='linear/linear_model/x1/weights')
        tf.Variable([0.2], name=BIAS_NAME)
        tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('x0'), self._fc_lib.numeric_column('x1')), model_dir=self._model_dir)
    predict_input_fn = numpy_io.numpy_input_fn(x={'x0': np.array([[2.0]]), 'x1': np.array([[3.0]])}, y=None, batch_size=1, num_epochs=1, shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    self.assertAllClose([[80.2]], predicted_scores)