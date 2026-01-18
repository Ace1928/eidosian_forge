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
def model_fn_inference_on_tpu(model_fn, features, labels=None, config=None, params=None, batch_config=None):
    """Convenience wrapper for export_saved_model API v2 for a model_fn.
  WARNING:THIS METHOD IS DEPRECATED AND NOT PART OF THE APIS.

  Make sure to set
  `export_saved_model_api_version=tpu_estimator.ExportSavedModelApiVersion.V2`
  when initializing TPUEstimator (default API version is V1). This is because
  1) `tpu.rewrite` (or `tpu.compile`) shouldn't be called in a nested way
      (otherwise validation will throw error like
      "NotImplementedError: tpu_shard_context cannot be nested.")
  2) When using V1 API, Estimator calls `tpu.rewrite` so
     using `model_fn_inference_on_tpu` will trigger a nested call.
     When using V2 API, users of Estimator needs to call `tpu.rewrite` (which
     the wrapper does).

  It attempts to execute the entire model function on the TPU for prediction.
  Note that this does not support features which are SparseTensors. If you have
  SparseTensor features, consider partitioning your model function further and
  use inference_on_tpu.

  Args:
    model_fn: the model_fn for which we want to inference on TPU.
    features: a tensor or dict of tensors, serves as the feature inputs to the
      model.
    labels: a tensor or dict of tensors, serves as the labels inputs to the
      model.
    config: auxiliary config to the Estimator.
    params: hparams that we want to pass to the model_fn.
    batch_config: a named tuple to wrap the inference batching configuration
      inputs.

  Returns:
    An EstimatorSpec containing the outputs in export_outputs and predictions.
  """
    computation, capture = _build_computation_for_inference(model_fn, labels, config, params)
    tensors = call_computation(features, computation, batch_config=batch_config)
    estimator_spec, export_outputs_dict, predictions_dict, none_indices = capture.get()
    predictions_list = tensors[:len(predictions_dict)]
    export_outputs_list_without_none = tensors[len(predictions_dict):]
    export_outputs_list = []
    while none_indices or export_outputs_list_without_none:
        if none_indices and none_indices[0] == len(export_outputs_list):
            export_outputs_list.append(None)
            none_indices.pop(0)
        else:
            export_outputs_list.append(export_outputs_list_without_none.pop(0))
    new_export_outputs_dict = tf.nest.pack_sequence_as(export_outputs_dict, export_outputs_list)
    export_outputs = estimator_spec.export_outputs
    new_export_outputs = collections.OrderedDict(((k, _clone_export_output_with_tensors(export_outputs[k], v)) for k, v in six.iteritems(new_export_outputs_dict)))
    new_predictions = tf.nest.pack_sequence_as(predictions_dict, predictions_list)
    if len(new_predictions) == 1 and _KEY_WHEN_PREDICTIONS_IS_A_TENSOR in new_predictions:
        new_predictions = new_predictions[_KEY_WHEN_PREDICTIONS_IS_A_TENSOR]
    return estimator_spec._replace(export_outputs=new_export_outputs, predictions=new_predictions)