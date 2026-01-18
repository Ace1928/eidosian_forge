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
def get_variable_layout(self, variable):
    variable_layout = self._layout_map[variable.path]
    if variable_layout is not None:
        return variable_layout
    variable_shard_spec = [None] * len(variable.shape)
    return TensorLayout(variable_shard_spec, self.device_mesh)