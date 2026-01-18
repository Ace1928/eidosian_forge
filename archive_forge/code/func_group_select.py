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
def group_select(selects, length=None, depth=None):
    """
    Given a list of key tuples to select, groups them into sensible
    chunks to avoid duplicating indexing operations.
    """
    if length is None and depth is None:
        length = depth = len(selects[0])
    getter = operator.itemgetter(depth - length)
    if length > 1:
        selects = sorted(selects, key=getter)
        grouped_selects = defaultdict(dict)
        for k, v in itertools.groupby(selects, getter):
            grouped_selects[k] = group_select(list(v), length - 1, depth)
        return grouped_selects
    else:
        return list(selects)