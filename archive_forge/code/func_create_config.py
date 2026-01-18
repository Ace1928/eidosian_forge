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
def create_config(self, base_config, obj):
    """Create a new SharedObjectConfig for a given object."""
    shared_object_config = SharedObjectConfig(base_config, self._next_id)
    self._next_id += 1
    try:
        self._shared_objects_config[obj] = shared_object_config
    except TypeError:
        pass
    return shared_object_config