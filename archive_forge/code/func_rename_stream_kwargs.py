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
def rename_stream_kwargs(stream, kwargs, reverse=False):
    """
    Given a stream and a kwargs dictionary of parameter values, map to
    the corresponding dictionary where the keys are substituted with the
    appropriately renamed string.

    If reverse, the output will be a dictionary using the original
    parameter names given a dictionary using the renamed equivalents.
    """
    mapped_kwargs = {}
    mapping = stream_name_mapping(stream, reverse=reverse)
    for k, v in kwargs.items():
        if k not in mapping:
            msg = 'Could not map key {key} {direction} renamed equivalent'
            direction = 'from' if reverse else 'to'
            raise KeyError(msg.format(key=repr(k), direction=direction))
        mapped_kwargs[mapping[k]] = v
    return mapped_kwargs