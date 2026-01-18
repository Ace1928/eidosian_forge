from __future__ import annotations
from collections.abc import (
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_hashable
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import SeriesApply
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow import (
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.sparse import SparseAccessor
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import (
from pandas.core.indexes.accessors import CombinedDatetimelikeProperties
from pandas.core.indexes.api import (
import pandas.core.indexes.base as ibase
from pandas.core.indexes.multi import maybe_droplevels
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.methods import selectn
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.core.strings.accessor import StringMethods
from pandas.core.tools.datetimes import to_datetime
import pandas.io.formats.format as fmt
from pandas.io.formats.info import (
import pandas.plotting
def case_when(self, caselist: list[tuple[ArrayLike | Callable[[Series], Series | np.ndarray | Sequence[bool]], ArrayLike | Scalar | Callable[[Series], Series | np.ndarray]],]) -> Series:
    """
        Replace values where the conditions are True.

        Parameters
        ----------
        caselist : A list of tuples of conditions and expected replacements
            Takes the form:  ``(condition0, replacement0)``,
            ``(condition1, replacement1)``, ... .
            ``condition`` should be a 1-D boolean array-like object
            or a callable. If ``condition`` is a callable,
            it is computed on the Series
            and should return a boolean Series or array.
            The callable must not change the input Series
            (though pandas doesn`t check it). ``replacement`` should be a
            1-D array-like object, a scalar or a callable.
            If ``replacement`` is a callable, it is computed on the Series
            and should return a scalar or Series. The callable
            must not change the input Series
            (though pandas doesn`t check it).

            .. versionadded:: 2.2.0

        Returns
        -------
        Series

        See Also
        --------
        Series.mask : Replace values where the condition is True.

        Examples
        --------
        >>> c = pd.Series([6, 7, 8, 9], name='c')
        >>> a = pd.Series([0, 0, 1, 2])
        >>> b = pd.Series([0, 3, 4, 5])

        >>> c.case_when(caselist=[(a.gt(0), a),  # condition, replacement
        ...                       (b.gt(0), b)])
        0    6
        1    3
        2    1
        3    2
        Name: c, dtype: int64
        """
    if not isinstance(caselist, list):
        raise TypeError(f'The caselist argument should be a list; instead got {type(caselist)}')
    if not caselist:
        raise ValueError('provide at least one boolean condition, with a corresponding replacement.')
    for num, entry in enumerate(caselist):
        if not isinstance(entry, tuple):
            raise TypeError(f'Argument {num} must be a tuple; instead got {type(entry)}.')
        if len(entry) != 2:
            raise ValueError(f'Argument {num} must have length 2; a condition and replacement; instead got length {len(entry)}.')
    caselist = [(com.apply_if_callable(condition, self), com.apply_if_callable(replacement, self)) for condition, replacement in caselist]
    default = self.copy()
    conditions, replacements = zip(*caselist)
    common_dtypes = [infer_dtype_from(arg)[0] for arg in [*replacements, default]]
    if len(set(common_dtypes)) > 1:
        common_dtype = find_common_type(common_dtypes)
        updated_replacements = []
        for condition, replacement in zip(conditions, replacements):
            if is_scalar(replacement):
                replacement = construct_1d_arraylike_from_scalar(value=replacement, length=len(condition), dtype=common_dtype)
            elif isinstance(replacement, ABCSeries):
                replacement = replacement.astype(common_dtype)
            else:
                replacement = pd_array(replacement, dtype=common_dtype)
            updated_replacements.append(replacement)
        replacements = updated_replacements
        default = default.astype(common_dtype)
    counter = reversed(range(len(conditions)))
    for position, condition, replacement in zip(counter, conditions[::-1], replacements[::-1]):
        try:
            default = default.mask(condition, other=replacement, axis=0, inplace=False, level=None)
        except Exception as error:
            raise ValueError(f'Failed to apply condition{position} and replacement{position}.') from error
    return default