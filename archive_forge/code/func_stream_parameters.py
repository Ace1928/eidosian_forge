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
def stream_parameters(streams, no_duplicates=True, exclude=None):
    """
    Given a list of streams, return a flat list of parameter name,
    excluding those listed in the exclude list.

    If no_duplicates is enabled, a KeyError will be raised if there are
    parameter name clashes across the streams.
    """
    if exclude is None:
        exclude = ['name', '_memoize_key']
    from ..streams import Params
    param_groups = {}
    for s in streams:
        if not s.contents and isinstance(s.hashkey, dict):
            param_groups[s] = list(s.hashkey)
        else:
            param_groups[s] = list(s.contents)
    if no_duplicates:
        seen, clashes = ({}, [])
        clash_streams = []
        for s in streams:
            if isinstance(s, Params):
                continue
            for c in param_groups[s]:
                if c in seen:
                    clashes.append(c)
                    if seen[c] not in clash_streams:
                        clash_streams.append(seen[c])
                    clash_streams.append(s)
                else:
                    seen[c] = s
        clashes = sorted(clashes)
        if clashes:
            clashing = ', '.join([repr(c) for c in clash_streams[:-1]])
            raise Exception(f'The supplied stream objects {clashing} and {clash_streams[-1]} clash on the following parameters: {clashes!r}')
    return [name for group in param_groups.values() for name in group if name not in exclude]