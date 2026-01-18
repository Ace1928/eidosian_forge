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
def testMultiDim(self):
    """Tests predict when all variables are multi-dimenstional."""
    batch_size = 2
    label_dimension = 3
    x_dim = 4
    feature_columns = (self._fc_lib.numeric_column('x', shape=(x_dim,)),)
    with tf.Graph().as_default():
        tf.Variable([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]], name='linear/linear_model/x/weights')
        tf.Variable([0.2, 0.4, 0.6], name=BIAS_NAME)
        tf.Variable(100, name='global_step', dtype=tf.dtypes.int64)
        save_variables_to_ckpt(self._model_dir)
    linear_regressor = self._linear_regressor_fn(feature_columns=feature_columns, label_dimension=label_dimension, model_dir=self._model_dir)
    predict_input_fn = numpy_io.numpy_input_fn(x={'x': np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])}, y=None, batch_size=batch_size, num_epochs=1, shuffle=False)
    predictions = linear_regressor.predict(input_fn=predict_input_fn)
    predicted_scores = list([x['predictions'] for x in predictions])
    self.assertAllClose([[30.2, 40.4, 50.6], [70.2, 96.4, 122.6]], predicted_scores)