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
def read_tensor_tracer_event_file(event_file):
    """Reads the event file written by tensor tracer.

  This can be used to read the full tensors written into binary event files by
  by TensorTracer with trace_mode=full_tensor_summary.

  Example usage:
    result_dict_list = tensor_tracer.read_tensor_tracer_event_file(
      event_file_path)
    for result_dict in result_dict_list:
      for step, tensor_dict in result_dict.items():
        for tensor_name, full_tensor_content in tensor_dict.items():
          logging.info(tensor_name, full_tensor_content)

  Args:
    event_file: Path to the event file that contains only tensor tracer events.
  Returns:
    A list of event dictionaries, each of which with the form:
    {step_number: {tensor_name: tensor_content}}. This is a list instead of
    a single event dictionary because it is possible that an event file may
    have multiple event traces, each of them covering the same step ranges.
  Raises:
    ValueError: If an unexpected trace is found.
  """
    step_occurrence_count = collections.defaultdict(int)
    step_occurrence_list = []
    for trace_event in summary_iterator.summary_iterator(event_file):
        if not trace_event.HasField('summary'):
            continue
        if len(trace_event.summary.value) != 1:
            raise ValueError('Single step contains %d summary values, expected 1.' % len(trace_event.summary.value))
        step = trace_event.step
        step_occurrence_count[step] += 1
        occurrence_idx = step_occurrence_count[step] - 1
        occurrence_size = len(step_occurrence_list)
        if occurrence_idx == occurrence_size:
            new_occurrence = collections.defaultdict(dict)
            step_occurrence_list.append(new_occurrence)
        elif occurrence_idx > occurrence_size:
            raise ValueError('Unexpected: occurrence_idx (%d) > occurrence_size (%d)' % (occurrence_idx, occurrence_size))
        tensor_value = trace_event.summary.value[0]
        tensor_name = tensor_value.tag
        real_shape = [d.size for d in tensor_value.tensor.tensor_shape.dim]
        tensor_content = np.frombuffer(tensor_value.tensor.tensor_content, dtypes.DType(tensor_value.tensor.dtype).as_numpy_dtype()).reshape(real_shape)
        step_occurrence_list[occurrence_idx][step][tensor_name] = tensor_content
    return step_occurrence_list