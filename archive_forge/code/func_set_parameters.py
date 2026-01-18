import collections
import hashlib
import operator
import os
import os.path
import sys
import numpy as np
from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util
def set_parameters(tensor_tracer_params=None):
    """Enables tensor tracer and sets its parameters.

  Example usage:
    tensor_tracer_parameters = {'trace_dir': '/usr/tmp/trace_dir',
                                'trace_mode': 'norm',
                                'report_file': '/usr/tmp/trace_dir/report.all'}
    tensor_tracer.set_parameters(tensor_tracer_parameters)

  This sets up the parameters for tensor tracer. A call to tensor tracer as
  below is necessary to enable debugging on CPUs and GPUs. On TPUs below can be
  skipped as this call is hooked into tpu.rewrite.
    tt = tensor_tracer.TensorTracer()
    loss = tt.trace_cpu(tf.get_default_graph(), tensor_fetches=loss)

  Args:
    tensor_tracer_params: Tensor tracer parameter dictionary. Below gives
    examples of these parameters: See tensor_tracer_report.py for all
      parameters.
        - enable: If set, tensor tracer will be enabled. Calling
          enable_tensor_tracer automatically adds this parameters.
        - trace_mode: The trace_mode to be used by tensor tracer. These include:
          - summary: Collects multiple statistics for traced tensors, and writes
            them a summary file that can be visualized using tensorboard. This
            mode currently only works for TPUEstimator. It can be also be used
            for other models, but outfeed must be handled by the user.
          - norm: Collects norm of each traced tensor and writes them into a
            text file pointed by 'trace_dir' flag. (Default mode).
          - nan-inf: Checks the existince of NaNs and Infs in the tensor, and
            writes a boolean value to a text file pointed by 'trace_dir' flag.
            Note that 'norm' mode can also capture this information with more
            numerical info.
          - max-abs: Collects the absolute max for each traced tensors and
            writes it into a text file pointed by 'trace_dir' flag.
          - full-tensor: Writes the full tensor content of the traced tensors
            into a text file pointed by 'trace_dir' flag.
          - part-tensor: Writes a part of the tensor content of the traced
            tensors into a text file pointed by 'trace_dir' flag.
          - full_tensor_summary: Writes the full tensors as binary event files.
            The outputs can be read using: trace =
              tensor_tracer.read_tensor_tracer_event_file(event_file_path)

        - report_file: Path to the metadata file that is written during graph
          construction. If not set, metadata will be printed to stdout during
          graph construction.
        - trace_dir: Path where the execution traces will be written during the
          graph execution. If not set, trace will be printed to stderr.
        - trace_level: Tensor tracer aims to trace everything it can. This
          introduces some overhead on graph execution and graph compilation
          times. Using trace_level parameter, it is possible to trace operation
          based on their priorities. For example, - trace_level=7 is the highest
          trace_level, in which every op is traced. - trace_level=6 will skip
          constant operations such as tf.constant. - trace_level=5 will skip
          less important ops such as tf.identities. - The default trace_level=3,
          that will skip concat ops, or random number generators. - To reduce
          the graph compile time overhead, trace_level can be set to 0, that
          will skip additions, and substractions, and multiplications as well.
        - excluded_opnames: If set, any matching op name will not be traced.
          excluded_opnames can be set as a regular expression. E.g,
          excluded_opnames=.* will exclude everything.
        - excluded_optypes: If set, any matching op type will not be traced.
          excluded_optypes can be set as a regular expression. E.g,
          excluded_optypes=.* will exclude everything. excluded_optypes=MatMul
          will exclude all MatMul ops from tracing.
        - included_opnames: If set, any matching op name will be forced to be
          traced. included_opnames can be set as a regular expression. E.g,
          '--included_opnames=some_op --excluded_opname=*.' will only trace
          some_op.
        - included_optypes: If set, any matching op type will be forced to be
          traced. included_optypes can be set as a regular expression. E.g,
          '--included_optypes=some_op_type --excluded_optypes=*.' will trace
          only the ops with type 'some_op_type'
        - flush_summaries: If summary mode is used, flush_summaries=1 will
          flush summaries using outside compilation. Note that, if used with
          low level APIs, flush_summaries=1 is necessary to obtain results.
        Advanced Flags:
        - trace_scalar: Scalar values are not traced by default. If this flag is
          set, scalar values will also be traced.
        - op_range: In the form of '%d:%d' that limits the tracing to the ops
          within this limit. --op_range='5:10' will trace only the ops that have
            topological order between 5-10.
        - submode: 'brief' or 'detailed'. If the trace mode is not compact,
          brief mode will print only the id of each traced tensor to save some
          space. 'detailed' mode prints the full tensor name.
        - use_fingerprint_subdirectory: The trace directory will be chosen as
          using the fingerprint of the trace metadata under the provided
          trace_dir.
  """
    enable_flags = '--%s=1' % tensor_tracer_flags.FLAG_NAME_ENABLE
    if tensor_tracer_params:
        for key, value in tensor_tracer_params.items():
            enable_flags += ' --%s=%s' % (key, value)
    os.environ[tensor_tracer_flags.FLAGS_ENV_VAR] = enable_flags