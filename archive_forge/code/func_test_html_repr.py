from __future__ import annotations
import numpy as np
import pytest
from packaging.version import parse as parse_version
import dask
import dask.array as da
from dask.array.reductions import nannumel, numel
from dask.array.utils import assert_eq
def test_html_repr():
    pytest.importorskip('jinja2')
    y = da.random.random((10, 10), chunks=(5, 5))
    y[y < 0.8] = 0
    y = y.map_blocks(sparse.COO.from_numpy)
    text = y._repr_html_()
    assert 'COO' in text
    assert 'sparse' in text
    assert 'Bytes' not in text