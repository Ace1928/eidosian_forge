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
@final
def infer_objects(self, copy: bool=True) -> Index:
    """
        If we have an object dtype, try to infer a non-object dtype.

        Parameters
        ----------
        copy : bool, default True
            Whether to make a copy in cases where no inference occurs.
        """
    if self._is_multi:
        raise NotImplementedError('infer_objects is not implemented for MultiIndex. Use index.to_frame().infer_objects() instead.')
    if self.dtype != object:
        return self.copy() if copy else self
    values = self._values
    values = cast('npt.NDArray[np.object_]', values)
    res_values = lib.maybe_convert_objects(values, convert_non_numeric=True)
    if copy and res_values is values:
        return self.copy()
    result = Index(res_values, name=self.name)
    if not copy and res_values is values and (self._references is not None):
        result._references = self._references
        result._references.add_index_reference(result)
    return result