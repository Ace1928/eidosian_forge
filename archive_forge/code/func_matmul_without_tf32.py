import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
def matmul_without_tf32(a, b, *args, **kwargs):
    """Run matmul but cast float32 inputs to float64 if TensorFloat-32 is enabled.

  This effectively runs matmul without TensorFloat-32. It should only be used in
  tests when verifying some other op or functions works correctly, e.g. to test
  `tf.linalg.sqrtm` by matrix multiplying the output of the op by itself. In
  such cases, the matmul itself is not being tested so it's OK to run it with
  higher precision.

  If a matmul itself is being tested, or some other op which uses matmul, use
  `run_without_tensor_float_32` instead.

  This also casts complex64 inputs to complex128, since TensorFloat-32 can also
  be used with complex64

  Args:
    a: First input to tf.linalg.matmul
    b: Second input to tf.linalg.matmul
    args: Other positional arguments to tf.linalg.matmul
    **kwargs: Other keyword arguments to tf.linalg.matmul

  Returns:
    A tensor with the same type as `a`.
  """
    if config.tensor_float_32_execution_enabled() and a.dtype == 'float32':
        a = math_ops.cast(a, 'float64')
        b = math_ops.cast(b, 'float64')
        ret = math_ops.matmul(a, b, *args, **kwargs)
        return math_ops.cast(ret, a.dtype)
    elif config.tensor_float_32_execution_enabled() and a.dtype == 'complex64':
        a = math_ops.cast(a, 'complex128')
        b = math_ops.cast(b, 'complex128')
        ret = math_ops.matmul(a, b, *args, **kwargs)
        return math_ops.cast(ret, a.dtype)
    else:
        return math_ops.matmul(a, b, *args, **kwargs)