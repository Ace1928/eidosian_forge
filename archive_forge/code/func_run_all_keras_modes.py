import collections
import functools
import itertools
import unittest
from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def run_all_keras_modes(test_or_class=None, config=None, always_skip_v1=False, always_skip_eager=False, **kwargs):
    """Execute the decorated test with all keras execution modes.

  This decorator is intended to be applied either to individual test methods in
  a `keras_parameterized.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times -
  once executing in legacy graph mode, once running eagerly and with
  `should_run_eagerly` returning True, and once running eagerly with
  `should_run_eagerly` returning False.

  If Tensorflow v2 behavior is enabled, legacy graph mode will be skipped, and
  the test will only run twice.

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_all_keras_modes
    def test_foo(self):
      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(
          optimizer, loss, metrics=metrics,
          run_eagerly=testing_utils.should_run_eagerly())

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test will try compiling & fitting the small functional mlp using all
  three Keras execution modes.

  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    config: An optional config_pb2.ConfigProto to use to configure the
      session when executing graphs.
    always_skip_v1: If True, does not try running the legacy graph mode even
      when Tensorflow v2 behavior is not enabled.
    always_skip_eager: If True, does not execute the decorated test
      with eager execution modes.
    **kwargs: Additional kwargs for configuring tests for
     in-progress Keras behaviors/ refactorings that we haven't fully
     rolled out yet

  Returns:
    Returns a decorator that will run the decorated test method multiple times.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
    if kwargs:
        raise ValueError('Unrecognized keyword args: {}'.format(kwargs))
    params = [('_v2_function', 'v2_function')]
    if not always_skip_eager:
        params.append(('_v2_eager', 'v2_eager'))
    if not (always_skip_v1 or tf2.enabled()):
        params.append(('_v1_session', 'v1_session'))

    def single_method_decorator(f):
        """Decorator that constructs the test cases."""

        @parameterized.named_parameters(*params)
        @functools.wraps(f)
        def decorated(self, run_mode, *args, **kwargs):
            """A run of a single test case w/ specified run mode."""
            if run_mode == 'v1_session':
                _v1_session_test(f, self, config, *args, **kwargs)
            elif run_mode == 'v2_eager':
                _v2_eager_test(f, self, *args, **kwargs)
            elif run_mode == 'v2_function':
                _v2_function_test(f, self, *args, **kwargs)
            else:
                return ValueError('Unknown run mode %s' % run_mode)
        return decorated
    return _test_or_class_decorator(test_or_class, single_method_decorator)