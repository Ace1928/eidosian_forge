from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import tensorflow as tf
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
def canonicalize_times_or_steps_from_output(times, steps, previous_model_output):
    """Canonicalizes either relative or absolute times, with error checking."""
    if steps is not None and times is not None:
        raise ValueError('Only one of `steps` and `times` may be specified.')
    if steps is None and times is None:
        raise ValueError('One of `steps` and `times` must be specified.')
    if times is not None:
        times = numpy.array(times)
        if len(times.shape) != 2:
            times = times[None, ...]
        if previous_model_output[feature_keys.FilteringResults.TIMES].shape[0] != times.shape[0]:
            raise ValueError('`times` must have a batch dimension matching the previous model output (got a batch dimension of {} for `times` and {} for the previous model output).'.format(times.shape[0], previous_model_output[feature_keys.FilteringResults.TIMES].shape[0]))
        if not (previous_model_output[feature_keys.FilteringResults.TIMES][:, -1] < times[:, 0]).all():
            raise ValueError('Prediction times must be after the corresponding previous model output.')
    if steps is not None:
        predict_times = previous_model_output[feature_keys.FilteringResults.TIMES][:, -1:] + 1 + numpy.arange(steps)[None, ...]
    else:
        predict_times = times
    return predict_times