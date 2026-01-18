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
def test_index_repr(self) -> None:
    from xarray.core.indexes import Index

    class CustomIndex(Index):
        names: tuple[str, ...]

        def __init__(self, names: tuple[str, ...]):
            self.names = names

        def __repr__(self):
            return f'CustomIndex(coords={self.names})'
    coord_names = ('x', 'y')
    index = CustomIndex(coord_names)
    names = ('x',)
    normal = formatting.summarize_index(names, index, col_width=20)
    assert names[0] in normal
    assert len(normal.splitlines()) == len(names)
    assert 'CustomIndex' in normal

    class IndexWithInlineRepr(CustomIndex):

        def _repr_inline_(self, max_width: int):
            return f'CustomIndex[{', '.join(self.names)}]'
    index = IndexWithInlineRepr(coord_names)
    inline = formatting.summarize_index(names, index, col_width=20)
    assert names[0] in inline
    assert index._repr_inline_(max_width=40) in inline