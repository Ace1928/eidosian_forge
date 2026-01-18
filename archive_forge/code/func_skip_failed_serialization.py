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
@tf_contextlib.contextmanager
def skip_failed_serialization():
    global _SKIP_FAILED_SERIALIZATION
    prev = _SKIP_FAILED_SERIALIZATION
    try:
        _SKIP_FAILED_SERIALIZATION = True
        yield
    finally:
        _SKIP_FAILED_SERIALIZATION = prev