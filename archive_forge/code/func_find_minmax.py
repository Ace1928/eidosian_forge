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
def find_minmax(lims, olims):
    """
    Takes (a1, a2) and (b1, b2) as input and returns
    (np.nanmin(a1, b1), np.nanmax(a2, b2)). Used to calculate
    min and max values of a number of items.
    """
    try:
        limzip = zip(list(lims), list(olims), [np.nanmin, np.nanmax])
        limits = tuple([float(fn([l, ol])) for l, ol, fn in limzip])
    except Exception:
        limits = (np.nan, np.nan)
    return limits