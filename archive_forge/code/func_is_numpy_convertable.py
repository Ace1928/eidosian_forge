import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
def is_numpy_convertable(v):
    """
    Return whether a value is meaningfully convertable to a numpy array
    via 'numpy.array'
    """
    return hasattr(v, '__array__') or hasattr(v, '__array_interface__')