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
@contextmanager
def external_write_mode(cls, file=None, nolock=False):
    """
        Disable tqdm within context and refresh tqdm when exits.
        Useful when writing to standard output stream
        """
    fp = file if file is not None else sys.stdout
    try:
        if not nolock:
            cls.get_lock().acquire()
        inst_cleared = []
        for inst in getattr(cls, '_instances', []):
            if hasattr(inst, 'start_t') and (inst.fp == fp or all((f in (sys.stdout, sys.stderr) for f in (fp, inst.fp)))):
                inst.clear(nolock=True)
                inst_cleared.append(inst)
        yield
        for inst in inst_cleared:
            inst.refresh(nolock=True)
    finally:
        if not nolock:
            cls._lock.release()