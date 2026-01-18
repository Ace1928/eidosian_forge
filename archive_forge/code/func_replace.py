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
def replace(self, **kwargs):
    if 'tpu_config' not in kwargs:
        return super(RunConfig, self).replace(**kwargs)
    tpu_config = kwargs.pop('tpu_config')
    new_instance = super(RunConfig, self).replace(**kwargs)
    new_instance._tpu_config = tpu_config
    return new_instance