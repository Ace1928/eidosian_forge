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
def run_in_graph_and_eager_modes(func=None, config=None, use_gpu=True, assert_no_eager_garbage=False):
    """Execute the decorated test with and without enabling eager execution.

  This function returns a decorator intended to be applied to test methods in
  a `tf.test.TestCase` class. Doing so will cause the contents of the test
  method to be executed twice - once normally, and once with eager execution
  enabled. This allows unittests to confirm the equivalence between eager
  and graph execution (see `tf.compat.v1.enable_eager_execution`).

  For example, consider the following unittest:

  ```python
  class MyTests(tf.test.TestCase):

    @run_in_graph_and_eager_modes
    def test_foo(self):
      x = tf.constant([1, 2])
      y = tf.constant([3, 4])
      z = tf.add(x, y)
      self.assertAllEqual([4, 6], self.evaluate(z))

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test validates that `tf.add()` has the same behavior when computed with
  eager execution enabled as it does when constructing a TensorFlow graph and
  executing the `z` tensor in a session.

  `deprecated_graph_mode_only`, `run_v1_only`, `run_v2_only`, and
  `run_in_graph_and_eager_modes` are available decorators for different
  v1/v2/eager/graph combinations.


  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    config: An optional config_pb2.ConfigProto to use to configure the session
      when executing graphs.
    use_gpu: If True, attempt to run as many operations as possible on GPU.
    assert_no_eager_garbage: If True, sets DEBUG_SAVEALL on the garbage
      collector and asserts that no extra garbage has been created when running
      the test with eager execution enabled. This will fail if there are
      reference cycles (e.g. a = []; a.append(a)). Off by default because some
      tests may create garbage for legitimate reasons (e.g. they define a class
      which inherits from `object`), and because DEBUG_SAVEALL is sticky in some
      Python interpreters (meaning that tests which rely on objects being
      collected elsewhere in the unit test file will not work). Additionally,
      checks that nothing still has a reference to Tensors that the test
      allocated.

  Returns:
    Returns a decorator that will run the decorated test method twice:
    once by constructing and executing a graph in a session and once with
    eager execution enabled.
  """

    def decorator(f):
        if tf_inspect.isclass(f):
            raise ValueError('`run_in_graph_and_eager_modes` only supports test methods. Did you mean to use `run_all_in_graph_and_eager_modes`?')

        def decorated(self, *args, **kwargs):
            logging.info('Running %s in GRAPH mode.', f.__name__)
            try:
                with context.graph_mode(), self.subTest('graph_mode'):
                    for class_ in self.__class__.mro():
                        if class_.__name__ == 'XLATestCase':
                            session_func = self.session
                            session_kwargs = {}
                            break
                    else:
                        session_func = self.test_session
                        session_kwargs = dict(use_gpu=use_gpu, config=config)
                    with session_func(**session_kwargs):
                        f(self, *args, **kwargs)
            except unittest.case.SkipTest:
                pass

            def run_eagerly(self, **kwargs):
                logging.info('Running %s in EAGER mode.', f.__name__)
                if not use_gpu:
                    with ops.device('/device:CPU:0'):
                        f(self, *args, **kwargs)
                else:
                    f(self, *args, **kwargs)
            if assert_no_eager_garbage:
                ops.reset_default_graph()
                run_eagerly = assert_no_new_tensors(assert_no_garbage_created(run_eagerly))
            self.tearDown()
            self._tempdir = None
            graph_for_eager_test = ops.Graph()
            with graph_for_eager_test.as_default(), context.eager_mode(), self.subTest('eager_mode'):
                self.setUp()
                run_eagerly(self, **kwargs)
            ops.dismantle_graph(graph_for_eager_test)
        return tf_decorator.make_decorator(f, decorated)
    if func is not None:
        return decorator(func)
    return decorator