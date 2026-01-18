from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register_lazy('rmm')
def register_rmm():
    import rmm
    if hasattr(rmm, 'DeviceBuffer'):

        @sizeof.register(rmm.DeviceBuffer)
        def sizeof_rmm_devicebuffer(x):
            return int(x.nbytes)