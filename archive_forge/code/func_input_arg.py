import inspect
from collections.abc import Iterable
from typing import Optional, Text
@property
def input_arg(self):
    """Name of the argument to be used as the input to the Keras
        `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__. Set to
        ``"inputs"``."""
    return self._input_arg