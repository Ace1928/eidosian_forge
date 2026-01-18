from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import os
import six
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
@property
def device_fn(self):
    """Returns the device_fn.

    If device_fn is not `None`, it overrides the default
    device function used in `Estimator`.
    Otherwise the default one is used.
    """
    return self._device_fn