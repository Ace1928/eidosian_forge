from __future__ import annotations
from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.core.dtypes.astype import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.io.formats.printing import (
def set_names(self, names, *, level=None, inplace: bool=False) -> Self | None:
    """
        Set Index or MultiIndex name.

        Able to set new names partially and by level.

        Parameters
        ----------

        names : label or list of label or dict-like for MultiIndex
            Name(s) to set.

            .. versionchanged:: 1.3.0

        level : int, label or list of int or label, optional
            If the index is a MultiIndex and names is not dict-like, level(s) to set
            (None for all levels). Otherwise level must be None.

            .. versionchanged:: 1.3.0

        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Index([1, 2, 3, 4], dtype='int64', name='quarter')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx = idx.set_names(['kind', 'year'])
        >>> idx.set_names('species', level=0)
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])

        When renaming levels with a dict, levels can not be passed.

        >>> idx.set_names({'kind': 'snake'})
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['snake', 'year'])
        """
    if level is not None and (not isinstance(self, ABCMultiIndex)):
        raise ValueError('Level must be None for non-MultiIndex')
    if level is not None and (not is_list_like(level)) and is_list_like(names):
        raise TypeError('Names must be a string when a single level is provided.')
    if not is_list_like(names) and level is None and (self.nlevels > 1):
        raise TypeError('Must pass list-like as `names`.')
    if is_dict_like(names) and (not isinstance(self, ABCMultiIndex)):
        raise TypeError('Can only pass dict-like as `names` for MultiIndex.')
    if is_dict_like(names) and level is not None:
        raise TypeError('Can not pass level for dictlike `names`.')
    if isinstance(self, ABCMultiIndex) and is_dict_like(names) and (level is None):
        level, names_adjusted = ([], [])
        for i, name in enumerate(self.names):
            if name in names.keys():
                level.append(i)
                names_adjusted.append(names[name])
        names = names_adjusted
    if not is_list_like(names):
        names = [names]
    if level is not None and (not is_list_like(level)):
        level = [level]
    if inplace:
        idx = self
    else:
        idx = self._view()
    idx._set_names(names, level=level)
    if not inplace:
        return idx
    return None