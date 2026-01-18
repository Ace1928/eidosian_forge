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
def make_path_unique(path, counts, new):
    """
    Given a path, a list of existing paths and counts for each of the
    existing paths.
    """
    added = False
    while any((path == c[:i] for c in counts for i in range(1, len(c) + 1))):
        count = counts[path]
        counts[path] += 1
        if not new and len(path) > 1 or added:
            path = path[:-1]
        else:
            added = True
        path = path + (int_to_roman(count),)
    if len(path) == 1:
        path = path + (int_to_roman(counts.get(path, 1)),)
    if path not in counts:
        counts[path] = 1
    return path