from __future__ import annotations
import copyreg
import multiprocessing
import multiprocessing.pool
import os
import pickle
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from warnings import warn
import cloudpickle
from dask import config
from dask.local import MultiprocessingPoolExecutor, get_async, reraise
from dask.optimization import cull, fuse
from dask.system import CPU_COUNT
from dask.typing import Key
from dask.utils import ensure_dict
def remote_exception(exc: Exception, tb) -> Exception:
    """Metaclass that wraps exception type in RemoteException"""
    if type(exc) in exceptions:
        typ = exceptions[type(exc)]
        return typ(exc, tb)
    else:
        try:
            typ = type(exc.__class__.__name__, (RemoteException, type(exc)), {'exception_type': type(exc)})
            exceptions[type(exc)] = typ
            return typ(exc, tb)
        except TypeError:
            return exc