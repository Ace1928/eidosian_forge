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
def arglexsort(arrays):
    """
    Returns the indices of the lexicographical sorting
    order of the supplied arrays.
    """
    dtypes = ','.join((array.dtype.str for array in arrays))
    recarray = np.empty(len(arrays[0]), dtype=dtypes)
    for i, array in enumerate(arrays):
        recarray[f'f{i}'] = array
    return recarray.argsort()