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
def set_description_str(self, desc=None, refresh=True):
    """Set/modify description without ': ' appended."""
    self.desc = desc or ''
    if refresh:
        self.refresh()