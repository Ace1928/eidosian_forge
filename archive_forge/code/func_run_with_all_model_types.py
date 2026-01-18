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
def run_with_all_model_types(test_or_class=None, exclude_models=None):
    """Execute the decorated test with all Keras model types.

  This decorator is intended to be applied either to individual test methods in
  a `keras_parameterized.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times - once
  for each Keras model type.

  The Keras model types are: ['functional', 'subclass', 'sequential']

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_with_all_model_types(
      exclude_models = ['sequential'])
    def test_foo(self):
      model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test tries building a small mlp as both a functional model and as a
  subclass model.

  We can also annotate the whole class if we want this to apply to all tests in
  the class:
  ```python
  @testing_utils.run_with_all_model_types(exclude_models = ['sequential'])
  class MyTests(testing_utils.KerasTestCase):

    def test_foo(self):
      model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```


  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    exclude_models: A collection of Keras model types to not run.
      (May also be a single model type not wrapped in a collection).
      Defaults to None.

  Returns:
    Returns a decorator that will run the decorated test method multiple times:
    once for each desired Keras model type.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
    model_types = ['functional', 'subclass', 'sequential']
    params = [('_%s' % model, model) for model in model_types if model not in nest.flatten(exclude_models)]

    def single_method_decorator(f):
        """Decorator that constructs the test cases."""

        @parameterized.named_parameters(*params)
        @functools.wraps(f)
        def decorated(self, model_type, *args, **kwargs):
            """A run of a single test case w/ the specified model type."""
            if model_type == 'functional':
                _test_functional_model_type(f, self, *args, **kwargs)
            elif model_type == 'subclass':
                _test_subclass_model_type(f, self, *args, **kwargs)
            elif model_type == 'sequential':
                _test_sequential_model_type(f, self, *args, **kwargs)
            else:
                raise ValueError('Unknown model type: %s' % (model_type,))
        return decorated
    return _test_or_class_decorator(test_or_class, single_method_decorator)