from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def standardize_sample_weights(x_weight, output_names):
    """Maps `sample_weight` or `class_weight` to model outputs.

  Args:
      x_weight: User-provided `sample_weight` or `class_weight` argument.
      output_names: List of output names (strings) in the model.

  Returns:
      A list of `sample_weight` or `class_weight` where there are exactly
          one element per model output.

  Raises:
      ValueError: In case of invalid user-provided argument.
  """
    if x_weight is None or (isinstance(x_weight, (list, tuple)) and len(x_weight) == 0):
        return [None for _ in output_names]
    if len(output_names) == 1:
        if isinstance(x_weight, (list, tuple)) and len(x_weight) == 1:
            return x_weight
        if isinstance(x_weight, dict) and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if isinstance(x_weight, (list, tuple)):
        if len(x_weight) != len(output_names):
            raise ValueError('Provided `sample_weights` was a list of ' + str(len(x_weight)) + ' elements, but the model has ' + str(len(output_names)) + ' outputs. You should provide one `sample_weights`array per model output.')
        return x_weight
    if isinstance(x_weight, collections.abc.Mapping):
        unknown = set(x_weight.keys()).difference(output_names)
        if unknown:
            raise ValueError('Unknown entries in sample_weights dictionary: {}. Only expected following keys: {}'.format(list(unknown), output_names))
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise TypeError('The model has multiple outputs, so `sample_weights` should be either a list or a dict. Provided `sample_weights` type not understood: ' + str(x_weight))