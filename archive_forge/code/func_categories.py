from __future__ import annotations
from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.accessor import (
from pandas.core.algorithms import (
from pandas.core.arrays._mixins import (
from pandas.core.base import (
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin
from pandas.io.formats import console
@property
def categories(self) -> Index:
    """
        The categories of this categorical.

        Setting assigns new values to each category (effectively a rename of
        each individual category).

        The assigned value has to be a list-like object. All items must be
        unique and the number of items in the new categories must be the same
        as the number of items in the old categories.

        Raises
        ------
        ValueError
            If the new categories do not validate as categories or if the
            number of new categories is unequal the number of old categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c', 'd'])
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.categories
        Index(['b', 'c', 'd'], dtype='object')

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.categories
        Index(['a', 'b'], dtype='object')

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'c', 'b', 'a', 'c', 'b'])
        >>> ci.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.categories
        Index(['c', 'b', 'a'], dtype='object')
        """
    return self.dtype.categories