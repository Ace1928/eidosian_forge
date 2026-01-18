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
def testDefaultPartitionerWithMultiplePsReplicas(self):
    partitions = 2
    x_dim = 32 << 20

    class FakeRunConfig(run_config.RunConfig):

        @property
        def num_ps_replicas(self):
            return partitions
    with tf.compat.v1.test.mock.patch.object(estimator, '_get_replica_device_setter', return_value=lambda _: '/cpu:0'):
        linear_regressor = self._linear_regressor_fn(feature_columns=(self._fc_lib.categorical_column_with_hash_bucket('language', hash_bucket_size=x_dim),), config=FakeRunConfig(), model_dir=self._model_dir)

        def _input_fn():
            return ({'language': tf.sparse.SparseTensor(values=['english', 'spanish'], indices=[[0, 0], [0, 1]], dense_shape=[1, 2])}, [[10.0]])
        hook = CheckPartitionerVarHook(self, LANGUAGE_WEIGHT_NAME, x_dim, partitions)
        linear_regressor.train(input_fn=_input_fn, steps=1, hooks=[hook])