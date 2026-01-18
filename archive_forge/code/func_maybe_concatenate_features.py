from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import enum
import math
import os
import signal
import sys
import threading
import time
import tensorflow as tf
import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.framework import variable_pb2
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf.tpu import compilation_result_pb2 as tpu_compilation_result
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import preempted_hook
from tensorflow.python.tpu import session_support
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_gradient
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import evaluation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_inspect
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output as export_output_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import error_handling
from tensorflow_estimator.python.estimator.tpu import iteration_count_estimator
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu import tpu_context
from tensorflow_estimator.python.estimator.tpu import util as util_lib
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdagradParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import AdamParameters  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import EmbeddingConfigSpec  # pylint: disable=unused-import
from tensorflow_estimator.python.estimator.tpu._tpu_estimator_embedding import StochasticGradientDescentParameters  # pylint: disable=unused-import
def maybe_concatenate_features(self, features):
    """If there are enough small tensors, concat them for performance."""
    self._small_feature_names = {}
    self._small_feature_sizes = {}
    feature_names = _extract_key_names(features)
    if feature_names:
        for name in feature_names:
            tensor = features[name]
            if not isinstance(tensor, tf.Tensor):
                return
            shape = tensor.get_shape().as_list()
            dtype = tensor.dtype
            if len(shape) == 2 and shape[1] is not None and (shape[1] <= self._small_feature_dim_size):
                tf.compat.v1.logging.log_first_n(tf.compat.v1.logging.INFO, 'Found small feature: %s %s', 1, name, shape)
                if tensor.dtype not in self._small_feature_names:
                    self._small_feature_names[dtype] = []
                    self._small_feature_sizes[dtype] = []
                self._small_feature_names[dtype].append(name)
                self._small_feature_sizes[dtype].append(shape[1])
        dtypes_ = list(self._small_feature_names.keys())
        for dtype in dtypes_:
            if len(self._small_feature_names[dtype]) < self._minimum_num_small_features_to_group:
                self._small_feature_names.pop(dtype)
                self._small_feature_sizes.pop(dtype)
        small_feature_tensors = {}
        for dtype in self._small_feature_names:
            small_feature_tensors[dtype] = []
            for name in self._small_feature_names[dtype]:
                small_feature_tensors[dtype].append(features.pop(name))
        for dtype in self._small_feature_names:
            key = self._get_small_feature_key(dtype)
            if key in features:
                raise ValueError('{} is reserved as feature key for concatenatedsmall features.')
            features[key] = tf.concat(small_feature_tensors[dtype], axis=1)