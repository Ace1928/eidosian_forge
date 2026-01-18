from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def build_supervised_input_receiver_fn_from_input_fn(input_fn, **input_fn_args):
    """Get a function that returns a SupervisedInputReceiver matching an input_fn.

  Note that this function calls the input_fn in a local graph in order to
  extract features and labels. Placeholders are then created from those
  features and labels in the default graph.

  Args:
    input_fn: An Estimator input_fn, which is a function that returns one of:
      * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a tuple
        (features, labels) with same constraints as below.
      * A tuple (features, labels): Where `features` is a `Tensor` or a
        dictionary of string feature name to `Tensor` and `labels` is a `Tensor`
        or a dictionary of string label name to `Tensor`. Both `features` and
        `labels` are consumed by `model_fn`. They should satisfy the expectation
        of `model_fn` from inputs.
    **input_fn_args: set of kwargs to be passed to the input_fn. Note that these
      will not be checked or validated here, and any errors raised by the
      input_fn will be thrown to the top.

  Returns:
    A function taking no arguments that, when called, returns a
    SupervisedInputReceiver. This function can be passed in as part of the
    input_receiver_map when exporting SavedModels from Estimator with multiple
    modes.
  """
    with tf.Graph().as_default():
        result = input_fn(**input_fn_args)
        features, labels, _ = util.parse_input_fn_result(result)
    return build_raw_supervised_input_receiver_fn(features, labels)