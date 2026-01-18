from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
def test_concat3():
    pdf1 = pd.DataFrame(np.random.randn(6, 5), columns=list('ABCDE'), index=list('abcdef'))
    pdf2 = pd.DataFrame(np.random.randn(6, 5), columns=list('ABCFG'), index=list('ghijkl'))
    pdf3 = pd.DataFrame(np.random.randn(6, 5), columns=list('ABCHI'), index=list('mnopqr'))
    ddf1 = dd.from_pandas(pdf1, 2)
    ddf2 = dd.from_pandas(pdf2, 3)
    ddf3 = dd.from_pandas(pdf3, 2)
    expected = pd.concat([pdf1, pdf2], sort=False)
    result = dd.concat([ddf1, ddf2])
    assert result.divisions == ddf1.divisions[:-1] + ddf2.divisions
    assert result.npartitions == ddf1.npartitions + ddf2.npartitions
    assert_eq(result, expected)
    assert_eq(dd.concat([ddf1, ddf2], interleave_partitions=True), pd.concat([pdf1, pdf2]))
    expected = pd.concat([pdf1, pdf2, pdf3], sort=False)
    result = dd.concat([ddf1, ddf2, ddf3])
    assert result.divisions == ddf1.divisions[:-1] + ddf2.divisions[:-1] + ddf3.divisions
    assert result.npartitions == ddf1.npartitions + ddf2.npartitions + ddf3.npartitions
    assert_eq(result, expected)
    assert_eq(dd.concat([ddf1, ddf2, ddf3], interleave_partitions=True), pd.concat([pdf1, pdf2, pdf3]))