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
def assert_no_garbage_created(f):
    """Test method decorator to assert that no garbage has been created.

  Note that this decorator sets DEBUG_SAVEALL, which in some Python interpreters
  cannot be un-set (i.e. will disable garbage collection for any other unit
  tests in the same file/shard).

  Args:
    f: The function to decorate.

  Returns:
    The decorated function.
  """

    def decorator(self, **kwargs):
        """Sets DEBUG_SAVEALL, runs the test, and checks for new garbage."""
        gc.disable()
        previous_debug_flags = gc.get_debug()
        gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
        gc.collect()
        previous_garbage = len(gc.garbage)
        result = f(self, **kwargs)
        gc.collect()
        new_garbage = len(gc.garbage)
        if new_garbage > previous_garbage:
            for i, obj in enumerate(gc.garbage[previous_garbage:]):
                if getattr(obj, '__module__', '') == 'ast':
                    new_garbage -= 3
        if new_garbage > previous_garbage:
            logging.error("The decorated test created work for Python's garbage collector, likely due to a reference cycle. New objects in cycle(s):")
            for i, obj in enumerate(gc.garbage[previous_garbage:]):
                try:
                    logging.error('Object %d of %d', i, len(gc.garbage) - previous_garbage)

                    def _safe_object_str(obj):
                        return '<%s %d>' % (obj.__class__.__name__, id(obj))
                    logging.error('  Object type: %s', _safe_object_str(obj))
                    logging.error('  Referrer types: %s', ', '.join([_safe_object_str(ref) for ref in gc.get_referrers(obj)]))
                    logging.error('  Referent types: %s', ', '.join([_safe_object_str(ref) for ref in gc.get_referents(obj)]))
                    logging.error('  Object attribute names: %s', dir(obj))
                    logging.error('  Object __str__:')
                    logging.error(obj)
                    logging.error('  Object __repr__:')
                    logging.error(repr(obj))
                except Exception:
                    logging.error('(Exception while printing object)')
        if new_garbage > previous_garbage:
            for i in range(previous_garbage, new_garbage):
                if _find_reference_cycle(gc.garbage, i):
                    break
        self.assertEqual(previous_garbage, new_garbage)
        gc.set_debug(previous_debug_flags)
        gc.enable()
        return result
    return decorator