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
@estimator_export('estimator.experimental.build_raw_supervised_input_receiver_fn')
def build_raw_supervised_input_receiver_fn(features, labels, default_batch_size=None):
    """Build a supervised_input_receiver_fn for raw features and labels.

  This function wraps tensor placeholders in a supervised_receiver_fn
  with the expectation that the features and labels appear precisely as
  the model_fn expects them. Features and labels can therefore be dicts of
  tensors, or raw tensors.

  Args:
    features: a dict of string to `Tensor` or `Tensor`.
    labels: a dict of string to `Tensor` or `Tensor`.
    default_batch_size: the number of query examples expected per batch. Leave
      unset for variable batch size (recommended).

  Returns:
    A supervised_input_receiver_fn.

  Raises:
    ValueError: if features and labels have overlapping keys.
  """
    try:
        feat_keys = features.keys()
    except AttributeError:
        feat_keys = [SINGLE_RECEIVER_DEFAULT_NAME]
    try:
        label_keys = labels.keys()
    except AttributeError:
        label_keys = [SINGLE_LABEL_DEFAULT_NAME]
    overlap_keys = set(feat_keys) & set(label_keys)
    if overlap_keys:
        raise ValueError('Features and labels must have distinct keys. Found overlapping keys: {}'.format(overlap_keys))

    def supervised_input_receiver_fn():
        """A receiver_fn that expects pass-through features and labels."""
        if not isinstance(features, dict):
            features_cp = _placeholder_from_tensor(features, default_batch_size)
            receiver_features = {SINGLE_RECEIVER_DEFAULT_NAME: features_cp}
        else:
            receiver_features = _placeholders_from_receiver_tensors_dict(features, default_batch_size)
            features_cp = receiver_features
        if not isinstance(labels, dict):
            labels_cp = _placeholder_from_tensor(labels, default_batch_size)
            receiver_labels = {SINGLE_LABEL_DEFAULT_NAME: labels_cp}
        else:
            receiver_labels = _placeholders_from_receiver_tensors_dict(labels, default_batch_size)
            labels_cp = receiver_labels
        receiver_tensors = dict(receiver_features)
        receiver_tensors.update(receiver_labels)
        return SupervisedInputReceiver(features_cp, labels_cp, receiver_tensors)
    return supervised_input_receiver_fn