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
def populate_dict_with_module_objects(target_dict, modules, obj_filter):
    for module in modules:
        for name in dir(module):
            obj = getattr(module, name)
            if obj_filter(obj):
                target_dict[name] = obj