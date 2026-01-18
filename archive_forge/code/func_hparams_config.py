import abc
import hashlib
import json
import random
import time
import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import metadata
from tensorboard.plugins.hparams import plugin_data_pb2
def hparams_config(hparams, metrics, time_created_secs=None):
    """Write a top-level experiment configuration.

    This configuration describes the hyperparameters and metrics that will
    be tracked in the experiment, but does not record any actual values of
    those hyperparameters and metrics. It can be created before any models
    are actually trained.

    Args:
      hparams: A list of `HParam` values.
      metrics: A list of `Metric` values.
      time_created_secs: The time that this experiment was created, as
        seconds since epoch. Defaults to the current time.

    Returns:
      A tensor whose value is `True` on success, or `False` if no summary
      was written because no default summary writer was available.
    """
    pb = hparams_config_pb(hparams=hparams, metrics=metrics, time_created_secs=time_created_secs)
    return _write_summary('hparams_config', pb)