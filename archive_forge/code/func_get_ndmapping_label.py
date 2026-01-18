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
def get_ndmapping_label(ndmapping, attr):
    """
    Function to get the first non-auxiliary object
    label attribute from an NdMapping.
    """
    label = None
    els = iter(ndmapping.data.values())
    while label is None:
        try:
            el = next(els)
        except StopIteration:
            return None
        if not getattr(el, '_auxiliary_component', True):
            label = getattr(el, attr)
    if attr == 'group':
        tp = type(el).__name__
        if tp == label:
            return None
    return label