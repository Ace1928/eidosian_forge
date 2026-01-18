from __future__ import annotations
import sys
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from numpy.core import defchararray
import xarray as xr
from xarray.core import formatting
from xarray.tests import requires_cftime, requires_dask, requires_netCDF4
@pytest.mark.parametrize('names', (('x',), ('x', 'y'), ('x', 'y', 'z'), ('x', 'y', 'z', 'a')))
def test_index_repr_grouping(self, names) -> None:
    from xarray.core.indexes import Index

    class CustomIndex(Index):

        def __init__(self, names):
            self.names = names

        def __repr__(self):
            return f'CustomIndex(coords={self.names})'
    index = CustomIndex(names)
    normal = formatting.summarize_index(names, index, col_width=20)
    assert all((name in normal for name in names))
    assert len(normal.splitlines()) == len(names)
    assert 'CustomIndex' in normal
    hint_chars = [line[2] for line in normal.splitlines()]
    if len(names) <= 1:
        assert hint_chars == [' ']
    else:
        assert hint_chars[0] == '┌' and hint_chars[-1] == '└'
        assert len(names) == 2 or hint_chars[1:-1] == ['│'] * (len(names) - 2)