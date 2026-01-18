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
def drop(self, codes, level: Index | np.ndarray | Iterable[Hashable] | None=None, errors: IgnoreRaise='raise') -> MultiIndex:
    """
        Make a new :class:`pandas.MultiIndex` with the passed list of codes deleted.

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples when ``level`` is not specified.
        level : int or level name, default None
        errors : str, default 'raise'

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=["number", "color"])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.drop([(1, 'green'), (2, 'purple')])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1, 'purple'),
                    (2,  'green')],
                   names=['number', 'color'])

        We can also drop from a specific level.

        >>> idx.drop('green', level='color')
        MultiIndex([(0, 'purple'),
                    (1, 'purple'),
                    (2, 'purple')],
                   names=['number', 'color'])

        >>> idx.drop([1, 2], level=0)
        MultiIndex([(0,  'green'),
                    (0, 'purple')],
                   names=['number', 'color'])
        """
    if level is not None:
        return self._drop_from_level(codes, level, errors)
    if not isinstance(codes, (np.ndarray, Index)):
        try:
            codes = com.index_labels_to_array(codes, dtype=np.dtype('object'))
        except ValueError:
            pass
    inds = []
    for level_codes in codes:
        try:
            loc = self.get_loc(level_codes)
            if isinstance(loc, int):
                inds.append(loc)
            elif isinstance(loc, slice):
                step = loc.step if loc.step is not None else 1
                inds.extend(range(loc.start, loc.stop, step))
            elif com.is_bool_indexer(loc):
                if self._lexsort_depth == 0:
                    warnings.warn('dropping on a non-lexsorted multi-index without a level parameter may impact performance.', PerformanceWarning, stacklevel=find_stack_level())
                loc = loc.nonzero()[0]
                inds.extend(loc)
            else:
                msg = f'unsupported indexer of type {type(loc)}'
                raise AssertionError(msg)
        except KeyError:
            if errors != 'ignore':
                raise
    return self.delete(inds)