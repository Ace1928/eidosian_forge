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
def cftime_to_timestamp(date, time_unit='us'):
    """Converts cftime to timestamp since epoch in milliseconds

    Non-standard calendars (e.g. Julian or no leap calendars)
    are converted to standard Gregorian calendar. This can cause
    extra space to be added for dates that don't exist in the original
    calendar. In order to handle these dates correctly a custom bokeh
    model with support for other calendars would have to be defined.

    Args:
        date: cftime datetime object (or array)

    Returns:
        time_unit since 1970-01-01 00:00:00
    """
    import cftime
    if time_unit == 'us':
        tscale = 1
    else:
        tscale = np.timedelta64(1, 'us') / np.timedelta64(1, time_unit)
    return cftime.date2num(date, 'microseconds since 1970-01-01 00:00:00', calendar='standard') * tscale