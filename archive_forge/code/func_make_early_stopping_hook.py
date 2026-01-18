import collections
import operator
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@estimator_export('estimator.experimental.make_early_stopping_hook')
def make_early_stopping_hook(estimator, should_stop_fn, run_every_secs=60, run_every_steps=None):
    """Creates early-stopping hook.

  Returns a `SessionRunHook` that stops training when `should_stop_fn` returns
  `True`.

  Usage example:

  ```python
  estimator = ...
  hook = early_stopping.make_early_stopping_hook(
      estimator, should_stop_fn=make_stop_fn(...))
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    should_stop_fn: `callable`, function that takes no arguments and returns a
      `bool`. If the function returns `True`, stopping will be initiated by the
      chief.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    A `SessionRunHook` that periodically executes `should_stop_fn` and initiates
    early stopping if the function returns `True`.

  Raises:
    TypeError: If `estimator` is not of type `tf.estimator.Estimator`.
    ValueError: If both `run_every_secs` and `run_every_steps` are set.
  """
    if not isinstance(estimator, estimator_lib.Estimator):
        raise TypeError('`estimator` must have type `tf.estimator.Estimator`. Got: {}'.format(type(estimator)))
    if run_every_secs is not None and run_every_steps is not None:
        raise ValueError('Only one of `run_every_secs` and `run_every_steps` must be set.')
    train_distribute = estimator.config.train_distribute
    mwms = ['CollectiveAllReduceStrategy', 'MultiWorkerMirroredStrategy']
    if train_distribute and (train_distribute.__class__.__name__.startswith(strategy) for strategy in mwms):
        if run_every_secs:
            raise ValueError('run_every_secs should not be set when using MultiWorkerMirroredStrategy.')
        return _MultiWorkerEarlyStoppingHook(should_stop_fn, run_every_steps)
    if estimator.config.is_chief:
        return _StopOnPredicateHook(should_stop_fn, run_every_secs, run_every_steps)
    else:
        return _CheckForStoppingHook()