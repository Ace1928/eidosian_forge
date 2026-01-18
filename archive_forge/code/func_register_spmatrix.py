from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register_lazy('scipy')
def register_spmatrix():
    import scipy
    from scipy import sparse
    if parse_version(scipy.__version__) < parse_version('1.12.0.dev0'):

        @sizeof.register(sparse.dok_matrix)
        def sizeof_spmatrix_dok(s):
            return s.__sizeof__()

    @sizeof.register(sparse.spmatrix)
    def sizeof_spmatrix(s):
        return sum((sizeof(v) for v in s.__dict__.values()))