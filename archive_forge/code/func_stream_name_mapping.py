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
def stream_name_mapping(stream, exclude_params=None, reverse=False):
    """
    Return a complete dictionary mapping between stream parameter names
    to their applicable renames, excluding parameters listed in
    exclude_params.

    If reverse is True, the mapping is from the renamed strings to the
    original stream parameter names.
    """
    if exclude_params is None:
        exclude_params = ['name']
    from ..streams import Params
    if isinstance(stream, Params):
        mapping = {}
        for p in stream.parameters:
            if isinstance(p, str):
                mapping[p] = stream._rename.get(p, p)
            else:
                mapping[p.name] = stream._rename.get((p.owner, p.name), p.name)
    else:
        filtered = [k for k in stream.param if k not in exclude_params]
        mapping = {k: stream._rename.get(k, k) for k in filtered}
    if reverse:
        return {v: k for k, v in mapping.items()}
    else:
        return mapping