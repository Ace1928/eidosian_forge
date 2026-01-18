import os
import sys
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.util import nest
def is_wrapped_model(layer):
    from tensorflow.python.keras.engine import functional
    from tensorflow.python.keras.layers import wrappers
    return isinstance(layer, wrappers.Wrapper) and isinstance(layer.layer, functional.Functional)