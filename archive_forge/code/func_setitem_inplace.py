from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def setitem_inplace(self, indexer, value, warn: bool=True) -> None:
    """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
    using_cow = using_copy_on_write()
    warn_cow = warn_copy_on_write()
    if (using_cow or warn_cow) and (not self._has_no_reference(0)):
        if using_cow:
            self.blocks = (self._block.copy(),)
            self._cache.clear()
        elif warn_cow and warn:
            warnings.warn(COW_WARNING_SETITEM_MSG, FutureWarning, stacklevel=find_stack_level())
    super().setitem_inplace(indexer, value)