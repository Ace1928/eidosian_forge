from __future__ import annotations
import bisect
import functools
import math
import warnings
from itertools import product
from numbers import Integral, Number
from operator import itemgetter
import numpy as np
from tlz import concat, memoize, merge, pluck
from dask import config, core, utils
from dask.array.chunk import getitem
from dask.base import is_dask_collection, tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.utils import cached_cumsum, is_arraylike
@functools.lru_cache
def block_index_from_1d_index(dim, loc0, loc1, is_bool):
    """The positions of index elements in the range values loc0 and loc1.

        The index is the input assignment index that is defined in the
        namespace of the caller. It is assumed that negative elements
        of an integer array have already been posified.

        The non-hashable dsk is the output dask dictionary that is
        defined in the namespace of the caller.

        Parameters
        ----------
        dim : `int`
           The dimension position of the index that is used as a proxy
           for the non-hashable index to define the LRU cache key.
        loc0 : `int`
            The start index of the block along the dimension.
        loc1 : `int`
            The stop index of the block along the dimension.
        is_bool : `bool`
            Whether or not the index is of boolean data type.

        Returns
        -------
        numpy array or `str`
            If index is a numpy array then a numpy array is
            returned.

            If index is a dask array then the dask of the block index
            is inserted into the output dask dictionary, and its
            unique top-layer key is returned.

        """
    if is_bool:
        i = index[loc0:loc1]
    elif is_dask_collection(index):
        i = np.where((loc0 <= index) & (index < loc1), index, loc1)
        i -= loc0
    else:
        i = np.where((loc0 <= index) & (index < loc1))[0]
        i = index[i] - loc0
    if is_dask_collection(i):
        i = concatenate_array_chunks(i)
        dsk.update(dict(i.dask))
        i = next(flatten(i.__dask_keys__()))
    return i