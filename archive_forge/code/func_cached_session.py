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
@contextlib.contextmanager
def cached_session(self, graph=None, config=None, use_gpu=True, force_gpu=False):
    """Returns a TensorFlow Session for use in executing tests.

    This method behaves differently than self.session(): for performance reasons
    `cached_session` will by default reuse the same session within the same
    test. The session returned by this function will only be closed at the end
    of the test (in the TearDown function).

    Use the `use_gpu` and `force_gpu` options to control where ops are run. If
    `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
    `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
    possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
    the CPU.

    Example:
    ```python
    class MyOperatorTest(test_util.TensorFlowTestCase):
      def testMyOperator(self):
        with self.cached_session() as sess:
          valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
          result = MyOperator(valid_input).eval()
          self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
          invalid_input = [-1.0, 2.0, 7.0]
          with self.assertRaisesOpError("negative input not supported"):
            MyOperator(invalid_input).eval()
    ```

    Args:
      graph: Optional graph to use during the returned session.
      config: An optional config_pb2.ConfigProto to use to configure the
        session.
      use_gpu: If True, attempt to run as many ops as possible on GPU.
      force_gpu: If True, pin all ops to `/device:GPU:0`.

    Yields:
      A Session object that should be used as a context manager to surround
      the graph building and execution code in a test case.
    """
    if context.executing_eagerly():
        yield FakeEagerSession(self)
    else:
        sess = self._get_cached_session(graph, config, force_gpu, crash_if_inconsistent_args=True)
        with self._constrain_devices_and_set_default(sess, use_gpu, force_gpu) as cached:
            yield cached