from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def set_flags(self, *, copy: bool_t=False, allows_duplicate_labels: bool_t | None=None) -> Self:
    """
        Return a new object with updated flags.

        Parameters
        ----------
        copy : bool, default False
            Specify if a copy of the object should be made.

            .. note::
                The `copy` keyword will change behavior in pandas 3.0.
                `Copy-on-Write
                <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
                will be enabled by default, which means that all methods with a
                `copy` keyword will use a lazy copy mechanism to defer the copy and
                ignore the `copy` keyword. The `copy` keyword will be removed in a
                future version of pandas.

                You can already get the future behavior and improvements through
                enabling copy on write ``pd.options.mode.copy_on_write = True``
        allows_duplicate_labels : bool, optional
            Whether the returned object allows duplicate labels.

        Returns
        -------
        Series or DataFrame
            The same type as the caller.

        See Also
        --------
        DataFrame.attrs : Global metadata applying to this dataset.
        DataFrame.flags : Global flags applying to this object.

        Notes
        -----
        This method returns a new object that's a view on the same data
        as the input. Mutating the input or the output values will be reflected
        in the other.

        This method is intended to be used in method chains.

        "Flags" differ from "metadata". Flags reflect properties of the
        pandas object (the Series or DataFrame). Metadata refer to properties
        of the dataset, and should be stored in :attr:`DataFrame.attrs`.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> df.flags.allows_duplicate_labels
        True
        >>> df2 = df.set_flags(allows_duplicate_labels=False)
        >>> df2.flags.allows_duplicate_labels
        False
        """
    df = self.copy(deep=copy and (not using_copy_on_write()))
    if allows_duplicate_labels is not None:
        df.flags['allows_duplicate_labels'] = allows_duplicate_labels
    return df