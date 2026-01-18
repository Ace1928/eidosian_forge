import logging
import threading
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import TensorStructType
from ray.rllib.utils.serialization import _serialize_ndarray, _deserialize_ndarray
from ray.rllib.utils.deprecation import deprecation_warning
@DeveloperAPI
def get_filter(filter_config, shape):
    if filter_config == 'MeanStdFilter':
        return MeanStdFilter(shape, clip=None)
    elif filter_config == 'ConcurrentMeanStdFilter':
        return ConcurrentMeanStdFilter(shape, clip=None)
    elif filter_config == 'NoFilter':
        return NoFilter()
    elif callable(filter_config):
        return filter_config(shape)
    else:
        raise Exception('Unknown observation_filter: ' + str(filter_config))