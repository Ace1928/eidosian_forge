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
def test_concat5():
    pdf1 = pd.DataFrame(np.random.randn(7, 5), columns=list('ABCDE'), index=list('abcdefg'))
    pdf2 = pd.DataFrame(np.random.randn(7, 6), columns=list('FGHIJK'), index=list('abcdefg'))
    pdf3 = pd.DataFrame(np.random.randn(7, 6), columns=list('FGHIJK'), index=list('cdefghi'))
    pdf4 = pd.DataFrame(np.random.randn(7, 5), columns=list('FGHAB'), index=list('cdefghi'))
    pdf5 = pd.DataFrame(np.random.randn(7, 5), columns=list('FGHAB'), index=list('fklmnop'))
    ddf1 = dd.from_pandas(pdf1, 2)
    ddf2 = dd.from_pandas(pdf2, 3)
    ddf3 = dd.from_pandas(pdf3, 2)
    ddf4 = dd.from_pandas(pdf4, 2)
    ddf5 = dd.from_pandas(pdf5, 3)
    cases = [[ddf1, ddf2], [ddf1, ddf3], [ddf1, ddf4], [ddf1, ddf5], [ddf3, ddf4], [ddf3, ddf5], [ddf5, ddf1, ddf4], [ddf5, ddf3], [ddf1.A, ddf4.A], [ddf2.F, ddf3.F], [ddf4.A, ddf5.A], [ddf1.A, ddf4.F], [ddf2.F, ddf3.H], [ddf4.A, ddf5.B], [ddf1, ddf4.A], [ddf3.F, ddf2], [ddf5, ddf1.A, ddf2]]
    for case in cases:
        pdcase = [c.compute() for c in case]
        assert_eq(dd.concat(case, interleave_partitions=True), pd.concat(pdcase, sort=False))
        assert_eq(dd.concat(case, join='inner', interleave_partitions=True), pd.concat(pdcase, join='inner'))
        assert_eq(dd.concat(case, axis=1), pd.concat(pdcase, axis=1))
        assert_eq(dd.concat(case, axis=1, join='inner'), pd.concat(pdcase, axis=1, join='inner'))
    cases = [[ddf1, pdf2], [ddf1, pdf3], [pdf1, ddf4], [pdf1.A, ddf4.A], [ddf2.F, pdf3.F], [ddf1, pdf4.A], [ddf3.F, pdf2], [ddf2, pdf1, ddf3.F]]
    for case in cases:
        if DASK_EXPR_ENABLED:
            from dask_expr._collection import FrameBase
            pdcase = [c.compute() if isinstance(c, FrameBase) else c for c in case]
        else:
            pdcase = [c.compute() if isinstance(c, _Frame) else c for c in case]
        assert_eq(dd.concat(case, interleave_partitions=True), pd.concat(pdcase))
        assert_eq(dd.concat(case, join='inner', interleave_partitions=True), pd.concat(pdcase, join='inner'))
        assert_eq(dd.concat(case, axis=1), pd.concat(pdcase, axis=1))
        assert_eq(dd.concat(case, axis=1, join='inner'), pd.concat(pdcase, axis=1, join='inner'))