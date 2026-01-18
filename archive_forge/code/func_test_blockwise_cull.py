from __future__ import annotations
import os
import threading
import xml.etree.ElementTree
from collections.abc import Set
from concurrent.futures import ThreadPoolExecutor
import pytest
import dask
from dask.base import tokenize
from dask.blockwise import Blockwise, blockwise_token
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer, to_graphviz
from dask.utils_test import inc
@pytest.mark.parametrize('flat', [True, False])
def test_blockwise_cull(flat):
    da = pytest.importorskip('dask.array')
    np = pytest.importorskip('numpy')
    if flat:
        x = da.from_array(np.arange(40).reshape((4, 10)), (2, 4)) + 100
    else:
        x = da.from_array(np.arange(10).reshape((10,)), (4,))
        y = da.from_array(np.arange(10).reshape((10,)), (4,))
        x = da.outer(x, y).transpose()
    dsk = x.__dask_graph__()
    select = (1, 1)
    keys = {(x._name, *select)}
    dsk_cull = dsk.cull(keys)
    for name, layer in dsk_cull.layers.items():
        if not isinstance(layer, dask.blockwise.Blockwise):
            assert not isinstance(dsk.layers[name], dask.blockwise.Blockwise)
            continue
        assert isinstance(dsk.layers[name], dask.blockwise.Blockwise)
        assert not layer.is_materialized()
        out_keys = layer.get_output_keys()
        assert out_keys == {(layer.output, *select)}
        assert not layer.is_materialized()