import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
@classmethod
def create_th_lock(cls):
    assert hasattr(cls, 'th_lock')
    warn('create_th_lock not needed anymore', TqdmDeprecationWarning, stacklevel=2)