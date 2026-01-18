import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def ensure_value_to_cell(value):
    """Ensures that a value is converted to a python cell object.

    Args:
        value: Any value that needs to be casted to the cell type

    Returns:
        A value wrapped as a cell object (see function "func_load")
    """

    def dummy_fn():
        value
    cell_value = dummy_fn.__closure__[0]
    if not isinstance(value, type(cell_value)):
        return cell_value
    return value