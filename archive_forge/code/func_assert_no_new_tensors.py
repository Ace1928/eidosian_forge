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
def assert_no_new_tensors(f):
    """Decorator for asserting that no new Tensors persist after a test.

  Mainly useful for checking that code using the Python C API has correctly
  manipulated reference counts.

  Clears the caches that it knows about, runs the garbage collector, then checks
  that there are no Tensor or Tensor-like objects still around. This includes
  Tensors to which something still has a reference (e.g. from missing
  Py_DECREFs) and uncollectable cycles (i.e. Python reference cycles where one
  of the objects has __del__ defined).

  Args:
    f: The test case to run.

  Returns:
    The decorated test case.
  """

    def decorator(self, **kwargs):
        """Finds existing Tensors, runs the test, checks for new Tensors."""

        def _is_tensorflow_object(obj):
            try:
                return isinstance(obj, (tensor_lib.Tensor, variables.Variable, tensor_shape.Dimension, tensor_shape.TensorShape))
            except (ReferenceError, AttributeError):
                return False
        tensors_before = set((id(obj) for obj in gc.get_objects() if _is_tensorflow_object(obj)))
        outside_executed_eagerly = context.executing_eagerly()
        outside_graph_key = ops.get_default_graph()._graph_key
        with ops.Graph().as_default():
            ops.get_default_graph()._graph_key = outside_graph_key
            if outside_executed_eagerly:
                with context.eager_mode():
                    result = f(self, **kwargs)
            else:
                result = f(self, **kwargs)
        context.context()._clear_caches()
        gc.collect()
        tensors_after = [obj for obj in gc.get_objects() if _is_tensorflow_object(obj) and id(obj) not in tensors_before]
        if tensors_after:
            raise AssertionError('%d Tensors not deallocated after test: %s' % (len(tensors_after), str(tensors_after)))
        return result
    return decorator