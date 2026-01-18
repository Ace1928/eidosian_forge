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
def reset_buffer(self) -> None:
    self.buffer = tree.map_structure(lambda s: RunningStat(s), self.shape)