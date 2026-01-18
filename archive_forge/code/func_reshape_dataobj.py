from __future__ import annotations
import typing as ty
import warnings
from contextlib import contextmanager
from threading import RLock
import numpy as np
from . import openers
from .fileslice import canonical_slicers, fileslice
from .volumeutils import apply_read_scaling, array_from_file
def reshape_dataobj(obj, shape):
    """Use `obj` reshape method if possible, else numpy reshape function"""
    return obj.reshape(shape) if hasattr(obj, 'reshape') else np.reshape(obj, shape)