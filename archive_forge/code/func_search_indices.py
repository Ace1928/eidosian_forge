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
def search_indices(values, source):
    """
    Given a set of values returns the indices of each of those values
    in the source array.
    """
    try:
        orig_indices = source.argsort()
    except TypeError:
        source = source.astype(str)
        values = values.astype(str)
        orig_indices = source.argsort()
    return orig_indices[np.searchsorted(source[orig_indices], values)]