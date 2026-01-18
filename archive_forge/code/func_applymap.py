from __future__ import annotations
import collections
from collections import abc
from collections.abc import (
import functools
from inspect import signature
from io import StringIO
import itertools
import operator
import sys
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import (
from pandas._config.config import _get_option
from pandas._libs import (
from pandas._libs.hashtable import duplicated
from pandas._libs.lib import is_range_indexer
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import (
from pandas.util._validators import (
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import CachedAccessor
from pandas.core.apply import reconstruct_and_relabel_result
from pandas.core.array_algos.take import take_2d_multi
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import (
from pandas.core.arrays.sparse import SparseFrameAccessor
from pandas.core.construction import (
from pandas.core.generic import (
from pandas.core.indexers import check_key_length
from pandas.core.indexes.api import (
from pandas.core.indexes.multi import (
from pandas.core.indexing import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods import selectn
from pandas.core.reshape.melt import melt
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import (
from pandas.io.common import get_handle
from pandas.io.formats import (
from pandas.io.formats.info import (
import pandas.plotting
def applymap(self, func: PythonFuncType, na_action: NaAction | None=None, **kwargs) -> DataFrame:
    """
        Apply a function to a Dataframe elementwise.

        .. deprecated:: 2.1.0

           DataFrame.applymap has been deprecated. Use DataFrame.map instead.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.
        **kwargs
            Additional keyword arguments to pass as keywords arguments to
            `func`.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        See Also
        --------
        DataFrame.apply : Apply a function along input axis of DataFrame.
        DataFrame.map : Apply a function along input axis of DataFrame.
        DataFrame.replace: Replace values given in `to_replace` with `value`.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> df.map(lambda x: len(str(x)))
           0  1
        0  3  4
        1  5  5
        """
    warnings.warn('DataFrame.applymap has been deprecated. Use DataFrame.map instead.', FutureWarning, stacklevel=find_stack_level())
    return self.map(func, na_action=na_action, **kwargs)