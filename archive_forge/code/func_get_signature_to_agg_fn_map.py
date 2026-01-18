import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def get_signature_to_agg_fn_map(self):
    """Returns a map that contains the aggregate function for each signature."""
    return {TRACE_MODE_NORM: linalg_ops.norm, TRACE_MODE_HISTORY: math_ops.reduce_max, TRACE_MODE_MAX_ABS: math_ops.reduce_max, TRACE_MODE_NAN_INF: math_ops.reduce_max, TT_SUMMARY_NORM: linalg_ops.norm, TT_SUMMARY_MAX: math_ops.reduce_max, TT_SUMMARY_MAX_ABS: lambda t, axis=0: math_ops.reduce_max(math_ops.abs(t), axis=axis), TT_SUMMARY_MIN: math_ops.reduce_min, TT_SUMMARY_SPARSITY: math_ops.reduce_mean, TT_SUMMARY_MEAN: math_ops.reduce_mean, TT_SUMMARY_VAR: math_ops.reduce_max, TT_SUMMARY_SIZE: math_ops.reduce_sum}