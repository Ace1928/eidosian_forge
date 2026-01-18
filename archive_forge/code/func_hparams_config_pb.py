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
def hparams_config_pb(hparams, metrics, time_created_secs=None):
    """Create a top-level experiment configuration.

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
      A TensorBoard `summary_pb2.Summary` message.
    """
    hparam_infos = []
    for hparam in hparams:
        info = api_pb2.HParamInfo(name=hparam.name, description=hparam.description, display_name=hparam.display_name)
        domain = hparam.domain
        if domain is not None:
            domain.update_hparam_info(info)
        hparam_infos.append(info)
    metric_infos = [metric.as_proto() for metric in metrics]
    experiment = api_pb2.Experiment(hparam_infos=hparam_infos, metric_infos=metric_infos, time_created_secs=time_created_secs)
    return _summary_pb(metadata.EXPERIMENT_TAG, plugin_data_pb2.HParamsPluginData(experiment=experiment))