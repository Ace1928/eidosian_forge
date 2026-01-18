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
@pytest.mark.skipif(PANDAS_GE_210, reason='breaks with pandas=2.1.0+')
@pytest.mark.slow
@pytest.mark.parametrize('how', ['inner', 'outer', 'left', 'right'])
def test_join_by_index_patterns(how, shuffle_method):

    def fix_index(out, dtype):
        if len(out) == 0:
            return out.set_index(out.index.astype(dtype))
        return out
    pdf1l = pd.DataFrame({'a': list('abcdefg'), 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf1r = pd.DataFrame({'c': list('abcdefg'), 'd': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf2l = pdf1l
    pdf2r = pd.DataFrame({'c': list('gfedcba'), 'd': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf3l = pdf1l
    pdf3r = pd.DataFrame({'c': list('abdg'), 'd': [5, 4, 3, 2]}, index=list('abdg'))
    pdf4l = pd.DataFrame({'a': list('abcabce'), 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf4r = pd.DataFrame({'c': list('abda'), 'd': [5, 4, 3, 2]}, index=list('abdg'))
    pdf5l = pd.DataFrame({'a': list('lmnopqr'), 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('lmnopqr'))
    pdf5r = pd.DataFrame({'c': list('abcd'), 'd': [5, 4, 3, 2]}, index=list('abcd'))
    pdf6l = pd.DataFrame({'a': list('cdefghi'), 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('cdefghi'))
    pdf6r = pd.DataFrame({'c': list('abab'), 'd': [5, 4, 3, 2]}, index=list('abcd'))
    pdf7l = pd.DataFrame({'a': list('aabbccd'), 'b': [7, 6, 5, 4, 3, 2, 1]}, index=list('abcdefg'))
    pdf7r = pd.DataFrame({'c': list('aabb'), 'd': [5, 4, 3, 2]}, index=list('fghi'))
    for pdl, pdr in [(pdf1l, pdf1r), (pdf2l, pdf2r), (pdf3l, pdf3r), (pdf4l, pdf4r), (pdf5l, pdf5r), (pdf6l, pdf6r), (pdf7l, pdf7r)]:
        for lpart, rpart in [(2, 2), (3, 2), (2, 3)]:
            ddl = dd.from_pandas(pdl, lpart)
            ddr = dd.from_pandas(pdr, rpart)
            assert_eq(ddl.join(ddr, how=how, shuffle_method=shuffle_method), fix_index(pdl.join(pdr, how=how), pdl.index.dtype))
            assert_eq(ddr.join(ddl, how=how, shuffle_method=shuffle_method), fix_index(pdr.join(pdl, how=how), pdr.index.dtype))
            assert_eq(ddl.join(ddr, how=how, lsuffix='l', rsuffix='r', shuffle_method=shuffle_method), fix_index(pdl.join(pdr, how=how, lsuffix='l', rsuffix='r'), pdl.index.dtype))
            assert_eq(ddr.join(ddl, how=how, lsuffix='l', rsuffix='r', shuffle_method=shuffle_method), fix_index(pdr.join(pdl, how=how, lsuffix='l', rsuffix='r'), pdl.index.dtype))
            list_eq(ddl.join(ddr, how=how, on='a', lsuffix='l', rsuffix='r'), pdl.join(pdr, how=how, on='a', lsuffix='l', rsuffix='r'))
            list_eq(ddr.join(ddl, how=how, on='c', lsuffix='l', rsuffix='r'), pdr.join(pdl, how=how, on='c', lsuffix='l', rsuffix='r'))
            list_eq(ddl.merge(ddr, how=how, left_on='a', right_index=True), pdl.merge(pdr, how=how, left_on='a', right_index=True))
            list_eq(ddr.merge(ddl, how=how, left_on='c', right_index=True), pdr.merge(pdl, how=how, left_on='c', right_index=True))
            list_eq(ddl.merge(ddr, how=how, left_index=True, right_on='c'), pdl.merge(pdr, how=how, left_index=True, right_on='c'))
            list_eq(ddr.merge(ddl, how=how, left_index=True, right_on='a'), pdr.merge(pdl, how=how, left_index=True, right_on='a'))