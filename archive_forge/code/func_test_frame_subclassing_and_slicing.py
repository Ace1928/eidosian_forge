import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_subclassing_and_slicing(self):

    class CustomSeries(Series):

        @property
        def _constructor(self):
            return CustomSeries

        def custom_series_function(self):
            return 'OK'

    class CustomDataFrame(DataFrame):
        """
            Subclasses pandas DF, fills DF with simulation results, adds some
            custom plotting functions.
            """

        def __init__(self, *args, **kw) -> None:
            super().__init__(*args, **kw)

        @property
        def _constructor(self):
            return CustomDataFrame
        _constructor_sliced = CustomSeries

        def custom_frame_function(self):
            return 'OK'
    data = {'col1': range(10), 'col2': range(10)}
    cdf = CustomDataFrame(data)
    assert isinstance(cdf, CustomDataFrame)
    cdf_series = cdf.col1
    assert isinstance(cdf_series, CustomSeries)
    assert cdf_series.custom_series_function() == 'OK'
    cdf_rows = cdf[1:5]
    assert isinstance(cdf_rows, CustomDataFrame)
    assert cdf_rows.custom_frame_function() == 'OK'
    mcol = MultiIndex.from_tuples([('A', 'A'), ('A', 'B')])
    cdf_multi = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
    assert isinstance(cdf_multi['A'], CustomDataFrame)
    mcol = MultiIndex.from_tuples([('A', ''), ('B', '')])
    cdf_multi2 = CustomDataFrame([[0, 1], [2, 3]], columns=mcol)
    assert isinstance(cdf_multi2['A'], CustomSeries)