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
def cross_index(values, index):
    """
    Allows efficiently indexing into a cartesian product without
    expanding it. The values should be defined as a list of iterables
    making up the cartesian product and a linear index, returning
    the cross product of the values at the supplied index.
    """
    lengths = [len(v) for v in values]
    length = np.prod(lengths)
    if index >= length:
        raise IndexError('Index %d out of bounds for cross-product of size %d' % (index, length))
    indexes = []
    for i in range(1, len(values))[::-1]:
        p = np.prod(lengths[-i:])
        indexes.append(index // p)
        index -= indexes[-1] * p
    indexes.append(index)
    return tuple((v[i] for v, i in zip(values, indexes)))