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
def generate_per_core_enqueue_ops_fn_for_host(ctx, input_fn, inputs_structure_recorder, host_device, host_id):
    """Generates infeed enqueue ops for per-core input_fn on a single host."""
    captured_infeed_queue = _CapturedObject()
    tpu_ordinal_function_impl = ctx.tpu_ordinal_function(host_id)

    def enqueue_ops_fn():
        """A fn returns enqueue_ops."""
        num_cores_per_host = ctx.num_of_cores_per_host
        per_host_sharded_inputs = []
        for core_ordinal in range(num_cores_per_host):
            with ops.name_scope('ordinal_%d' % core_ordinal):
                user_context = tpu_context.TPUContext(internal_ctx=ctx, input_device=host_device, invocation_index=host_id * ctx.num_of_cores_per_host + core_ordinal, host_id=host_id)
                inputs = _Inputs.from_input_fn(input_fn(user_context))
                if inputs.is_dataset:
                    raise TypeError('`input_fn` returning `Dataset`  is not yet supported in per-Core input pipeline deployment yet. Please set TPUConfig.per_host_input_for_training to True or return `features` and `labels` from `input_fn`')
                features, labels = inputs.features_and_labels()
                inputs_structure_recorder.validate_and_record_structure(features, labels)
                flattened_inputs = inputs_structure_recorder.flatten_features_and_labels(features, labels)
                per_host_sharded_inputs.append(flattened_inputs)
        infeed_queue = tpu_feed.InfeedQueue(number_of_tuple_elements=len(per_host_sharded_inputs[0]))
        captured_infeed_queue.capture(infeed_queue)
        per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(per_host_sharded_inputs, tpu_ordinal_function=tpu_ordinal_function_impl)
        return per_host_enqueue_ops
    return (enqueue_ops_fn, captured_infeed_queue)