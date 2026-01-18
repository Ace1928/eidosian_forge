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
@Appender(_index_shared_docs['get_indexer_non_unique'] % _index_doc_kwargs)
def get_indexer_non_unique(self, target) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    target = ensure_index(target)
    target = self._maybe_cast_listlike_indexer(target)
    if not self._should_compare(target) and (not self._should_partial_index(target)):
        return self._get_indexer_non_comparable(target, method=None, unique=False)
    pself, ptarget = self._maybe_downcast_for_indexing(target)
    if pself is not self or ptarget is not target:
        return pself.get_indexer_non_unique(ptarget)
    if self.dtype != target.dtype:
        dtype = self._find_common_type_compat(target)
        this = self.astype(dtype, copy=False)
        that = target.astype(dtype, copy=False)
        return this.get_indexer_non_unique(that)
    if self._is_multi and target._is_multi:
        engine = self._engine
        tgt_values = engine._extract_level_codes(target)
    else:
        tgt_values = target._get_engine_target()
    indexer, missing = self._engine.get_indexer_non_unique(tgt_values)
    return (ensure_platform_int(indexer), ensure_platform_int(missing))