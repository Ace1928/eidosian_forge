from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def maybe_overwrite_model_dir_and_session_config(config, model_dir):
    """Overwrite estimator config by `model_dir` and `session_config` if needed.

  Args:
    config: Original estimator config.
    model_dir: Estimator model checkpoint directory.

  Returns:
    Overwritten estimator config.

  Raises:
    ValueError: Model directory inconsistent between `model_dir` and `config`.
  """
    if config is None:
        config = run_config.RunConfig()
        tf.compat.v1.logging.info('Using default config.')
    if not isinstance(config, run_config.RunConfig):
        raise ValueError('config must be an instance of `RunConfig`, but provided %s.' % config)
    if config.session_config is None:
        session_config = run_config.get_default_session_config()
        config = run_config.RunConfig.replace(config, session_config=session_config)
    model_dir = run_config.path_to_str(model_dir)
    if model_dir is not None:
        if getattr(config, 'model_dir', None) is not None and config.model_dir != model_dir:
            raise ValueError("`model_dir` are set both in constructor and `RunConfig`, but with different values. In constructor: '{}', in `RunConfig`: '{}' ".format(model_dir, config.model_dir))
    if model_dir:
        config = run_config.RunConfig.replace(config, model_dir=model_dir)
    elif getattr(config, 'model_dir', None) is None:
        model_dir = tempfile.mkdtemp()
        tf.compat.v1.logging.warn('Using temporary folder as model directory: %s', model_dir)
        config = run_config.RunConfig.replace(config, model_dir=model_dir)
    return config