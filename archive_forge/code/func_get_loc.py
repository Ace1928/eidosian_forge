from __future__ import annotations
from collections.abc import (
from functools import wraps
from sys import getsizeof
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import coerce_indexer_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_array_like
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import validate_putmask
from pandas.core.arrays import (
from pandas.core.arrays.categorical import (
import pandas.core.common as com
from pandas.core.construction import sanitize_array
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.io.formats.printing import (
def get_loc(self, key):
    """
        Get location for a label or a tuple of labels.

        The location is returned as an integer/slice or boolean
        mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)

        Returns
        -------
        int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Notes
        -----
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_loc('b')
        slice(1, 3, None)

        >>> mi.get_loc(('b', 'e'))
        1
        """
    self._check_indexing_error(key)

    def _maybe_to_slice(loc):
        """convert integer indexer to boolean mask or slice if possible"""
        if not isinstance(loc, np.ndarray) or loc.dtype != np.intp:
            return loc
        loc = lib.maybe_indices_to_slice(loc, len(self))
        if isinstance(loc, slice):
            return loc
        mask = np.empty(len(self), dtype='bool')
        mask.fill(False)
        mask[loc] = True
        return mask
    if not isinstance(key, tuple):
        loc = self._get_level_indexer(key, level=0)
        return _maybe_to_slice(loc)
    keylen = len(key)
    if self.nlevels < keylen:
        raise KeyError(f'Key length ({keylen}) exceeds index depth ({self.nlevels})')
    if keylen == self.nlevels and self.is_unique:
        try:
            return self._engine.get_loc(key)
        except KeyError as err:
            raise KeyError(key) from err
        except TypeError:
            loc, _ = self.get_loc_level(key, list(range(self.nlevels)))
            return loc
    i = self._lexsort_depth
    lead_key, follow_key = (key[:i], key[i:])
    if not lead_key:
        start = 0
        stop = len(self)
    else:
        try:
            start, stop = self.slice_locs(lead_key, lead_key)
        except TypeError as err:
            raise KeyError(key) from err
    if start == stop:
        raise KeyError(key)
    if not follow_key:
        return slice(start, stop)
    warnings.warn('indexing past lexsort depth may impact performance.', PerformanceWarning, stacklevel=find_stack_level())
    loc = np.arange(start, stop, dtype=np.intp)
    for i, k in enumerate(follow_key, len(lead_key)):
        mask = self.codes[i][loc] == self._get_loc_single_level_index(self.levels[i], k)
        if not mask.all():
            loc = loc[mask]
        if not len(loc):
            raise KeyError(key)
    return _maybe_to_slice(loc) if len(loc) != stop - start else slice(start, stop)