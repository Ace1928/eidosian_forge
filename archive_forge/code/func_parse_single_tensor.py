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
def parse_single_tensor(x, tensor_name):
    if not isinstance(x, dict):
        return {tensor_name: x}
    elif len(x) == 1 and tensor_name == list(x.keys())[0]:
        return x
    else:
        raise PredictionError(PredictionError.INVALID_INPUTS, 'Expected tensor name: %s, got tensor name: %s.' % (tensor_name, list(x.keys())))