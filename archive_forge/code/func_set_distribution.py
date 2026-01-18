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
@keras_export('keras.distribution.set_distribution')
def set_distribution(value):
    """Set the distribution as the global distribution setting.

    Args:
        value: a `Distribution` instance.
    """
    global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, value)