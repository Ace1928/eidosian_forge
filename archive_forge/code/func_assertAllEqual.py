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
@py_func_if_in_function
def assertAllEqual(self, a, b, msg=None):
    """Asserts that two numpy arrays or Tensors have the same values.

    Args:
      a: the expected numpy ndarray or anything can be converted to one.
      b: the actual numpy ndarray or anything can be converted to one.
      msg: Optional message to report on failure.
    """
    if ragged_tensor.is_ragged(a) or ragged_tensor.is_ragged(b):
        return self._assertRaggedEqual(a, b, msg)
    msg = msg if msg else ''
    a, b = self.evaluate_if_both_tensors(a, b)
    a = self._GetNdArray(a)
    b = self._GetNdArray(b)
    if b.ndim <= 3 or b.size < 500:
        self.assertEqual(a.shape, b.shape, 'Shape mismatch: expected %s, got %s. Contents: %r. \n%s.' % (a.shape, b.shape, b, msg))
    else:
        self.assertEqual(a.shape, b.shape, 'Shape mismatch: expected %s, got %s. %s' % (a.shape, b.shape, msg))
    same = a == b
    if dtypes.as_dtype(a.dtype).is_floating:
        same = np.logical_or(same, np.logical_and(np.isnan(a), np.isnan(b)))
    msgs = [msg]
    if not np.all(same):
        diff = np.logical_not(same)
        if a.ndim:
            x = a[np.where(diff)]
            y = b[np.where(diff)]
            msgs.append('not equal where = {}'.format(np.where(diff)))
        else:
            x, y = (a, b)
        msgs.append('not equal lhs = %r' % x)
        msgs.append('not equal rhs = %r' % y)
        if a.dtype.kind != b.dtype.kind and {a.dtype.kind, b.dtype.kind}.issubset({'U', 'S', 'O'}):
            a_list = []
            b_list = []
            for out_list, flat_arr in [(a_list, a.flat), (b_list, b.flat)]:
                for item in flat_arr:
                    if isinstance(item, str):
                        out_list.append(item.encode('utf-8'))
                    else:
                        out_list.append(item)
            a = np.array(a_list)
            b = np.array(b_list)
        np.testing.assert_array_equal(a, b, err_msg='\n'.join(msgs))