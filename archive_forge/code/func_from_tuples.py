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
@classmethod
@names_compat
def from_tuples(cls, tuples: Iterable[tuple[Hashable, ...]], sortorder: int | None=None, names: Sequence[Hashable] | Hashable | None=None) -> MultiIndex:
    """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> pd.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
    if not is_list_like(tuples):
        raise TypeError('Input must be a list / sequence of tuple-likes.')
    if is_iterator(tuples):
        tuples = list(tuples)
    tuples = cast(Collection[tuple[Hashable, ...]], tuples)
    if len(tuples) and all((isinstance(e, tuple) and (not e) for e in tuples)):
        codes = [np.zeros(len(tuples))]
        levels = [Index(com.asarray_tuplesafe(tuples, dtype=np.dtype('object')))]
        return cls(levels=levels, codes=codes, sortorder=sortorder, names=names, verify_integrity=False)
    arrays: list[Sequence[Hashable]]
    if len(tuples) == 0:
        if names is None:
            raise TypeError('Cannot infer number of levels from empty list')
        arrays = [[]] * len(names)
    elif isinstance(tuples, (np.ndarray, Index)):
        if isinstance(tuples, Index):
            tuples = np.asarray(tuples._values)
        arrays = list(lib.tuples_to_object_array(tuples).T)
    elif isinstance(tuples, list):
        arrays = list(lib.to_object_array_tuples(tuples).T)
    else:
        arrs = zip(*tuples)
        arrays = cast(list[Sequence[Hashable]], arrs)
    return cls.from_arrays(arrays, sortorder=sortorder, names=names)