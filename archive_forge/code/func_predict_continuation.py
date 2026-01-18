from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys as _feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as _head
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils as _model_utils
def predict_continuation(continue_from, signatures, session, steps=None, times=None, exogenous_features=None):
    """Perform prediction using an exported saved model.

  Args:
    continue_from: A dictionary containing the results of either an Estimator's
      evaluate method or filter_continuation. Used to determine the model state
      to make predictions starting from.
    signatures: The `MetaGraphDef` protocol buffer returned from
      `tf.saved_model.loader.load`. Used to determine the names of Tensors to
      feed and fetch. Must be from the same model as `continue_from`.
    session: The session to use. The session's graph must be the one into which
      `tf.saved_model.loader.load` loaded the model.
    steps: The number of steps to predict (scalar), starting after the
      evaluation or filtering. If `times` is specified, `steps` must not be; one
      is required.
    times: A [batch_size x window_size] array of integers (not a Tensor)
      indicating times to make predictions for. These times must be after the
      corresponding evaluation or filtering. If `steps` is specified, `times`
      must not be; one is required. If the batch dimension is omitted, it is
      assumed to be 1.
    exogenous_features: Optional dictionary. If specified, indicates exogenous
      features for the model to use while making the predictions. Values must
      have shape [batch_size x window_size x ...], where `batch_size` matches
      the batch dimension used when creating `continue_from`, and `window_size`
      is either the `steps` argument or the `window_size` of the `times`
      argument (depending on which was specified).

  Returns:
    A dictionary with model-specific predictions (typically having keys "mean"
    and "covariance") and a _feature_keys.PredictionResults.TIMES key indicating
    the times for which the predictions were computed.
  Raises:
    ValueError: If `times` or `steps` are misspecified.
  """
    if exogenous_features is None:
        exogenous_features = {}
    predict_times = _model_utils.canonicalize_times_or_steps_from_output(times=times, steps=steps, previous_model_output=continue_from)
    features = {_feature_keys.PredictionFeatures.TIMES: predict_times}
    features.update(exogenous_features)
    predict_signature = signatures.signature_def[_feature_keys.SavedModelLabels.PREDICT]
    output_tensors_by_name, feed_dict = _colate_features_to_feeds_and_fetches(continue_from=continue_from, signature=predict_signature, features=features, graph=session.graph)
    output = session.run(output_tensors_by_name, feed_dict=feed_dict)
    output[_feature_keys.PredictionResults.TIMES] = features[_feature_keys.PredictionFeatures.TIMES]
    return output