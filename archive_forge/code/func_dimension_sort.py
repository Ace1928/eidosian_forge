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
def dimension_sort(odict, kdims, vdims, key_index):
    """
    Sorts data by key using usual Python tuple sorting semantics
    or sorts in categorical order for any categorical Dimensions.
    """
    sortkws = {}
    ndims = len(kdims)
    dimensions = kdims + vdims
    indexes = [(dimensions[i], int(i not in range(ndims)), i if i in range(ndims) else i - ndims) for i in key_index]
    cached_values = {d.name: [None] + list(d.values) for d in dimensions}
    if len(set(key_index)) != len(key_index):
        raise ValueError('Cannot sort on duplicated dimensions')
    else:
        sortkws['key'] = lambda x: tuple((cached_values[dim.name].index(x[t][d]) if dim.values else x[t][d] for i, (dim, t, d) in enumerate(indexes)))
    return python2sort(odict.items(), **sortkws)