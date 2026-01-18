from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.slow
def test_arithmetics_different_index():
    pdf1 = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [3, 5, 2, 5, 7]}, index=[1, 2, 3, 4, 5])
    ddf1 = dd.from_pandas(pdf1, 2)
    pdf2 = pd.DataFrame({'a': [3, 2, 6, 7, 8], 'b': [9, 4, 2, 6, 2]}, index=[3, 4, 5, 6, 7])
    ddf2 = dd.from_pandas(pdf2, 2)
    pdf3 = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [3, 5, 2, 5, 7]}, index=[1, 2, 3, 4, 5])
    ddf3 = dd.from_pandas(pdf3, 2)
    pdf4 = pd.DataFrame({'a': [3, 2, 6, 7, 8], 'b': [9, 4, 2, 6, 2]}, index=[10, 11, 12, 13, 14])
    ddf4 = dd.from_pandas(pdf4, 2)
    pdf5 = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [3, 5, 2, 5, 7]}, index=[1, 3, 5, 7, 9])
    ddf5 = dd.from_pandas(pdf5, 2)
    pdf6 = pd.DataFrame({'a': [3, 2, 6, 7, 8], 'b': [9, 4, 2, 6, 2]}, index=[2, 3, 4, 5, 6])
    ddf6 = dd.from_pandas(pdf6, 2)
    cases = [(ddf1, ddf2, pdf1, pdf2), (ddf2, ddf1, pdf2, pdf1), (ddf1.repartition([1, 3, 5]), ddf2.repartition([3, 4, 7]), pdf1, pdf2), (ddf2.repartition([3, 4, 5, 7]), ddf1.repartition([1, 2, 4, 5]), pdf2, pdf1), (ddf3, ddf4, pdf3, pdf4), (ddf4, ddf3, pdf4, pdf3), (ddf3.repartition([1, 2, 3, 4, 5]), ddf4.repartition([10, 11, 12, 13, 14]), pdf3, pdf4), (ddf4.repartition([10, 14]), ddf3.repartition([1, 3, 4, 5]), pdf4, pdf3), (ddf5, ddf6, pdf5, pdf6), (ddf6, ddf5, pdf6, pdf5), (ddf5.repartition([1, 7, 8, 9]), ddf6.repartition([2, 3, 4, 6]), pdf5, pdf6), (ddf6.repartition([2, 6]), ddf5.repartition([1, 3, 7, 9]), pdf6, pdf5), (ddf1, pdf2, pdf1, pdf2), (ddf2, pdf1, pdf2, pdf1), (ddf3, pdf4, pdf3, pdf4), (ddf4, pdf3, pdf4, pdf3), (ddf5, pdf6, pdf5, pdf6), (ddf6, pdf5, pdf6, pdf5)]
    for l, r, el, er in cases:
        check_series_arithmetics(l.a, r.b, el.a, er.b, allow_comparison_ops=False)
        check_frame_arithmetics(l, r, el, er, allow_comparison_ops=False)
    pdf7 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8], 'b': [5, 6, 7, 8, 1, 2, 3, 4]}, index=[0, 2, 4, 8, 9, 10, 11, 13])
    pdf8 = pd.DataFrame({'a': [5, 6, 7, 8, 4, 3, 2, 1], 'b': [2, 4, 5, 3, 4, 2, 1, 0]}, index=[1, 3, 4, 8, 9, 11, 12, 13])
    ddf7 = dd.from_pandas(pdf7, 3)
    ddf8 = dd.from_pandas(pdf8, 2)
    pdf9 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8], 'b': [5, 6, 7, 8, 1, 2, 3, 4]}, index=[0, 2, 4, 8, 9, 10, 11, 13])
    pdf10 = pd.DataFrame({'a': [5, 6, 7, 8, 4, 3, 2, 1], 'b': [2, 4, 5, 3, 4, 2, 1, 0]}, index=[0, 3, 4, 8, 9, 11, 12, 13])
    ddf9 = dd.from_pandas(pdf9, 3)
    ddf10 = dd.from_pandas(pdf10, 2)
    cases = [(ddf7, ddf8, pdf7, pdf8), (ddf8, ddf7, pdf8, pdf7), (ddf8.repartition([-5, 10, 15], force=True), ddf7.repartition([-1, 4, 11, 14], force=True), pdf8, pdf7), (ddf7.repartition([0, 8, 12, 13]), ddf8.repartition([0, 2, 8, 12, 13], force=True), pdf7, pdf8), (ddf8.repartition([-5, 0, 10, 20], force=True), ddf7.repartition([-1, 4, 11, 13], force=True), pdf8, pdf7), (ddf9, ddf10, pdf9, pdf10), (ddf10, ddf9, pdf10, pdf9), (ddf7, pdf8, pdf7, pdf8), (ddf8, pdf7, pdf8, pdf7), (ddf9, pdf10, pdf9, pdf10), (ddf10, pdf9, pdf10, pdf9)]
    for l, r, el, er in cases:
        check_series_arithmetics(l.a, r.b, el.a, er.b, allow_comparison_ops=False)
        check_frame_arithmetics(l, r, el, er, allow_comparison_ops=False)