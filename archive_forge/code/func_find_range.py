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
def find_range(values, soft_range=None):
    """
    Safely finds either the numerical min and max of
    a set of values, falling back to the first and
    the last value in the sorted list of values.
    """
    if soft_range is None:
        soft_range = []
    try:
        values = np.array(values)
        values = np.squeeze(values) if len(values.shape) > 1 else values
        if len(soft_range):
            values = np.concatenate([values, soft_range])
        if values.dtype.kind == 'M':
            return (values.min(), values.max())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
            return (np.nanmin(values), np.nanmax(values))
    except Exception:
        try:
            values = sorted(values)
            return (values[0], values[-1])
        except Exception:
            return (None, None)