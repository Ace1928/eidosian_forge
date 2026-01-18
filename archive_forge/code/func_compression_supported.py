from ray.rllib.utils.annotations import DeveloperAPI
import logging
import time
import base64
import numpy as np
from ray import cloudpickle as pickle
@DeveloperAPI
def compression_supported():
    return LZ4_ENABLED