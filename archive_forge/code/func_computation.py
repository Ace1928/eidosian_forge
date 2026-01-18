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
def computation():
    """Compute tpu tensors used in export_outputs.

    Passed to rewrite_for_inference so that model_fn will be called under
    the rewriting contexts. Only tpu tensors are returned, but export_outputs
    and scaffold are captured.

    Returns:
       A list of Tensors used in export_outputs and not marked for
       outside_compilation.
    """
    model_fn_args = function_utils.fn_args(model_fn)
    kwargs = {}
    if 'labels' in model_fn_args:
        kwargs['labels'] = labels
    if 'mode' in model_fn_args:
        kwargs['mode'] = model_fn_lib.ModeKeys.PREDICT
    if 'config' in model_fn_args:
        kwargs['config'] = config
    if 'params' in model_fn_args:
        kwargs['params'] = params
    estimator_spec = model_fn(features, **kwargs)
    export_outputs_dict = collections.OrderedDict(((k, _export_output_to_tensors(v)) for k, v in six.iteritems(estimator_spec.export_outputs)))
    export_outputs_list = tf.nest.flatten(export_outputs_dict)
    export_outputs_tpu_list = [t for t in export_outputs_list if t is not None]
    if isinstance(estimator_spec.predictions, dict):
        predictions_dict = collections.OrderedDict(((k, v) for k, v in six.iteritems(estimator_spec.predictions)))
    else:
        predictions_dict = {_KEY_WHEN_PREDICTIONS_IS_A_TENSOR: estimator_spec.predictions}
    predictions_list = tf.nest.flatten(predictions_dict)
    capture.capture((estimator_spec, export_outputs_dict, export_outputs_list, predictions_dict))
    return predictions_list + export_outputs_tpu_list