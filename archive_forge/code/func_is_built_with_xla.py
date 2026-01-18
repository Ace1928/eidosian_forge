from tensorflow.python.framework import test_util as _test_util
from tensorflow.python.platform import googletest as _googletest
from tensorflow.python.framework.test_util import assert_equal_graph_def
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.framework.test_util import is_gpu_available
from tensorflow.python.ops.gradient_checker import compute_gradient_error
from tensorflow.python.ops.gradient_checker import compute_gradient
import functools
import sys
from tensorflow.python.util.tf_export import tf_export
@tf_export('test.is_built_with_xla')
def is_built_with_xla():
    """Returns whether TensorFlow was built with XLA support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with XLA.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_xla(self):
  ...     if not tf.test.is_built_with_xla():
  ...       self.skipTest("test is only applicable on XLA")

  ...     @tf.function(jit_compile=True)
  ...     def add(x, y):
  ...       return tf.math.add(x, y)
  ...
  ...     self.assertEqual(add(tf.ones(()), tf.ones(())), 2.0)

  TensorFlow official binary is built with XLA.
  """
    return _test_util.IsBuiltWithXLA()