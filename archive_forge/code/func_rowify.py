import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
def rowify(columns):
    """Converts columnar input to row data.

  Consider the following code:

    columns = {"prediction": np.array([1,             # 1st instance
                                       0,             # 2nd
                                       1]),           # 3rd
               "scores": np.array([[0.1, 0.9],        # 1st instance
                                   [0.7, 0.3],        # 2nd
                                   [0.4, 0.6]])}      # 3rd

  Then rowify will return the equivalent of:

    [{"prediction": 1, "scores": [0.1, 0.9]},
     {"prediction": 0, "scores": [0.7, 0.3]},
     {"prediction": 1, "scores": [0.4, 0.6]}]

  (each row is yielded; no list is actually created).

  Arguments:
    columns: (dict) mapping names to numpy arrays, where the arrays
      contain a batch of data.

  Raises:
    PredictionError: if the outer dimension of each input isn't identical
    for each of element.

  Yields:
    A map with a single instance, as described above. Note: instances
    is not a numpy array.
  """
    sizes_set = {e.shape[0] for e in six.itervalues(columns)}
    if len(sizes_set) != 1:
        sizes_dict = {name: e.shape[0] for name, e in six.iteritems(columns)}
        raise PredictionError(PredictionError.INVALID_OUTPUTS, 'Bad output from running tensorflow session: outputs had differing sizes in the batch (outer) dimension. See the outputs and their size: %s. Check your model for bugs that effect the size of the outputs.' % sizes_dict)
    num_instances = len(next(six.itervalues(columns)))
    for row in six.moves.xrange(num_instances):
        yield {name: output[row, ...].tolist() for name, output in six.iteritems(columns)}