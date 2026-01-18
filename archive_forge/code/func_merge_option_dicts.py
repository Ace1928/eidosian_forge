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
def merge_option_dicts(old_opts, new_opts):
    """
    Update the old_opts option dictionary with the options defined in
    new_opts. Instead of a shallow update as would be performed by calling
    old_opts.update(new_opts), this updates the dictionaries of all option
    types separately.

    Given two dictionaries
        old_opts = {'a': {'x': 'old', 'y': 'old'}}
    and
        new_opts = {'a': {'y': 'new', 'z': 'new'}, 'b': {'k': 'new'}}
    this returns a dictionary
        {'a': {'x': 'old', 'y': 'new', 'z': 'new'}, 'b': {'k': 'new'}}
    """
    merged = dict(old_opts)
    for option_type, options in new_opts.items():
        if option_type not in merged:
            merged[option_type] = {}
        merged[option_type].update(options)
    return merged