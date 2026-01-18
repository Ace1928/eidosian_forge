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
def canonicalize_single_tensor_input(instances, tensor_name):
    """Canonicalize single input tensor instances into list of dicts.

  Instances that are single input tensors may or may not be provided with their
  tensor name. The following are both valid instances:
    1) instances = [{"x": "a"}, {"x": "b"}, {"x": "c"}]
    2) instances = ["a", "b", "c"]
  This function canonicalizes the input instances to be of type 1).

  Arguments:
    instances: single input tensor instances as supplied by the user to the
      predict method.
    tensor_name: the expected name of the single input tensor.

  Raises:
    PredictionError: if the wrong tensor name is supplied to instances.

  Returns:
    A list of dicts. Where each dict is a single instance, mapping the
    tensor_name to the value (as supplied by the original instances).
  """

    def parse_single_tensor(x, tensor_name):
        if not isinstance(x, dict):
            return {tensor_name: x}
        elif len(x) == 1 and tensor_name == list(x.keys())[0]:
            return x
        else:
            raise PredictionError(PredictionError.INVALID_INPUTS, 'Expected tensor name: %s, got tensor name: %s.' % (tensor_name, list(x.keys())))
    if not isinstance(instances, list):
        instances = [instances]
    instances = [parse_single_tensor(x, tensor_name) for x in instances]
    return instances