import collections
import contextlib
import os
import re
import warnings
import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state
def get_data_layout(self, data_shape):
    data_shard_spec = [None] * len(data_shape)
    data_shard_spec[0] = self._batch_dim_name
    return TensorLayout(data_shard_spec, self.device_mesh)