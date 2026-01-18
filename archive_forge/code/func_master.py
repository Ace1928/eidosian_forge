from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.tpu import util as util_lib
@property
def master(self):
    return self._master