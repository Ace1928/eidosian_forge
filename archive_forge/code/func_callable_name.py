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
def callable_name(callable_obj):
    """
    Attempt to return a meaningful name identifying a callable or generator
    """
    try:
        if isinstance(callable_obj, type) and issubclass(callable_obj, param.ParameterizedFunction):
            return callable_obj.__name__
        elif isinstance(callable_obj, param.Parameterized) and 'operation' in callable_obj.param:
            return callable_obj.operation.__name__
        elif isinstance(callable_obj, partial):
            return str(callable_obj)
        elif inspect.isfunction(callable_obj):
            return callable_obj.__name__
        elif inspect.ismethod(callable_obj):
            return callable_obj.__func__.__qualname__.replace('.__call__', '')
        elif isinstance(callable_obj, types.GeneratorType):
            return callable_obj.__name__
        else:
            return type(callable_obj).__name__
    except Exception:
        return str(callable_obj)