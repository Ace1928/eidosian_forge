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
@classmethod
def from_client(cls, client, unused_model_path, **unused_kwargs):
    """Creates a TensorFlowModel from a SessionClient and model data files."""
    return cls(client)