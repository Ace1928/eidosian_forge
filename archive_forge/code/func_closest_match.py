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
def closest_match(match, specs, depth=0):
    """
    Recursively iterates over type, group, label and overlay key,
    finding the closest matching spec.
    """
    if len(match) == 0:
        return None
    new_specs = []
    match_lengths = []
    for i, spec in specs:
        if spec[0] == match[0]:
            new_specs.append((i, spec[1:]))
        else:
            if all((isinstance(s[0], str) for s in [spec, match])):
                match_length = max((i for i in range(len(match[0])) if match[0].startswith(spec[0][:i])))
            elif is_number(match[0]) and is_number(spec[0]):
                m = bool(match[0]) if isinstance(match[0], np.bool_) else match[0]
                s = bool(spec[0]) if isinstance(spec[0], np.bool_) else spec[0]
                match_length = -abs(m - s)
            else:
                match_length = 0
            match_lengths.append((i, match_length, spec[0]))
    if len(new_specs) == 1:
        return new_specs[0][0]
    elif new_specs:
        depth = depth + 1
        return closest_match(match[1:], new_specs, depth)
    elif depth == 0 or not match_lengths:
        return None
    else:
        return sorted(match_lengths, key=lambda x: -x[1])[0][0]