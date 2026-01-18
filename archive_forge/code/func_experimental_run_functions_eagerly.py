from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Use `tf.config.run_functions_eagerly` instead of the experimental version.')
@tf_export('config.experimental_run_functions_eagerly')
def experimental_run_functions_eagerly(run_eagerly):
    """Enables / disables eager execution of `tf.function`s.

  Calling `tf.config.experimental_run_functions_eagerly(True)` will make all
  invocations of `tf.function` run eagerly instead of running as a traced graph
  function.

  See `tf.config.run_functions_eagerly` for an example.

  Note: This flag has no effect on functions passed into tf.data transformations
  as arguments. tf.data functions are never executed eagerly and are always
  executed as a compiled Tensorflow Graph.

  Args:
    run_eagerly: Boolean. Whether to run functions eagerly.

  Returns:
    None
  """
    return run_functions_eagerly(run_eagerly)