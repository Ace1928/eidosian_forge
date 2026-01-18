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
def iterative_select(obj, dimensions, selects, depth=None):
    """
    Takes the output of group_select selecting subgroups iteratively,
    avoiding duplicating select operations.
    """
    ndims = len(dimensions)
    depth = depth if depth is not None else ndims
    items = []
    if isinstance(selects, dict):
        for k, v in selects.items():
            items += iterative_select(obj.select(**{dimensions[ndims - depth]: k}), dimensions, v, depth - 1)
    else:
        for s in selects:
            items.append((s, obj.select(**{dimensions[-1]: s[-1]})))
    return items