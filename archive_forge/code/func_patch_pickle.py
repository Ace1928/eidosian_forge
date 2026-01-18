from __future__ import annotations
import contextlib
import copy
import io
import pickle as pkl
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset
from pandas import Index
from pandas.core.arrays import (
from pandas.core.internals import BlockManager
@contextlib.contextmanager
def patch_pickle() -> Generator[None, None, None]:
    """
    Temporarily patch pickle to use our unpickler.
    """
    orig_loads = pkl.loads
    try:
        setattr(pkl, 'loads', loads)
        yield
    finally:
        setattr(pkl, 'loads', orig_loads)