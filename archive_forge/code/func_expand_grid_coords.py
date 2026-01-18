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
def expand_grid_coords(dataset, dim):
    """
    Expand the coordinates along a dimension of the gridded
    dataset into an ND-array matching the dimensionality of
    the dataset.
    """
    irregular = [d.name for d in dataset.kdims if d is not dim and dataset.interface.irregular(dataset, d)]
    if irregular:
        array = dataset.interface.coords(dataset, dim, True)
        example = dataset.interface.values(dataset, irregular[0], True, False)
        return array * np.ones_like(example)
    else:
        arrays = [dataset.interface.coords(dataset, d.name, True) for d in dataset.kdims]
        idx = dataset.get_dimension_index(dim)
        return cartesian_product(arrays, flat=False)[idx].T