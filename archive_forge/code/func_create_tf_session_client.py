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
def create_tf_session_client(model_dir, tags=(SERVING,), config=None):
    return SessionClient(*load_tf_model(model_dir, tags, config))