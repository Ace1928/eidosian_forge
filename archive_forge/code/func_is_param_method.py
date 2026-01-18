import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def is_param_method(obj, has_deps=False):
    """Whether the object is a method on a parameterized object.

    Args:
       obj: Object to check
       has_deps (boolean, optional): Check for dependencies
          Whether to also check whether the method has been annotated
          with param.depends

    Returns:
       A boolean value indicating whether the object is a method
       on a Parameterized object and if enabled whether it has any
       dependencies
    """
    parameterized = inspect.ismethod(obj) and isinstance(get_method_owner(obj), param.Parameterized)
    if parameterized and has_deps:
        return getattr(obj, '_dinfo', {}).get('dependencies')
    return parameterized