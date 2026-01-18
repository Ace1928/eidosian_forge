from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.experimental.call_logit_fn')
def call_logit_fn(logit_fn, features, mode, params, config):
    """Calls logit_fn (experimental).

  THIS FUNCTION IS EXPERIMENTAL. Keras layers/models are the recommended APIs
  for logit and model composition.

  A utility function that calls the provided logit_fn with the relevant subset
  of provided arguments. Similar to tf.estimator._call_model_fn().

  Args:
    logit_fn: A logit_fn as defined above.
    features: The features dict.
    mode: TRAIN / EVAL / PREDICT ModeKeys.
    params: The hyperparameter dict.
    config: The configuration object.

  Returns:
    A logit Tensor, the output of logit_fn.

  Raises:
    ValueError: if logit_fn does not return a Tensor or a dictionary mapping
      strings to Tensors.
  """
    logit_fn_args = function_utils.fn_args(logit_fn)
    kwargs = {}
    if 'mode' in logit_fn_args:
        kwargs['mode'] = mode
    if 'params' in logit_fn_args:
        kwargs['params'] = params
    if 'config' in logit_fn_args:
        kwargs['config'] = config
    logit_fn_results = logit_fn(features=features, **kwargs)
    result_is_valid_dictionary = isinstance(logit_fn_results, dict) and all([isinstance(k, six.string_types) and isinstance(v, tf.Tensor) for k, v in six.iteritems(logit_fn_results)])
    result_is_tensor = isinstance(logit_fn_results, tf.Tensor)
    if not (result_is_valid_dictionary or result_is_tensor):
        raise ValueError('logit_fn should return a Tensor or a dictionary mapping strings to Tensors.  logit_fn returned: %s' % logit_fn_results)
    return logit_fn_results