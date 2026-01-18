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
def testFromScratch(self):
    label = 5.0
    age = 17
    mock_optimizer = self._mock_optimizer(expected_loss=25.0)
    linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.numeric_column('age'),), model_dir=self._model_dir, optimizer=mock_optimizer)
    self.assertEqual(0, mock_optimizer.minimize.call_count)
    num_steps = 10
    linear_regressor.train(input_fn=lambda: ({'age': ((age,),)}, ((label,),)), steps=num_steps)
    self.assertEqual(1, mock_optimizer.minimize.call_count)
    self._assert_checkpoint(expected_global_step=num_steps, expected_age_weight=0.0, expected_bias=0.0)