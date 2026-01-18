from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
@property
def current_host(self):
    """The current host index for the TPU system.

    Returns:
      The host index (int).

    Raises:
      RuntimeError: If this method is not be called from input_fn.
    """
    if not self._call_from_input_fn:
        raise RuntimeError('This TPUContext instance must not be called from model_fn.')
    return self._host_id