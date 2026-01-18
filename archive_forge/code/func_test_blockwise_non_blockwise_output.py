from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc
def test_blockwise_non_blockwise_output():
    x = da.ones(10, chunks=(5,))
    y = x + 1 + 2 + 3
    w = y.sum()
    z = y * 2 * 3 * 4
    z_top_before = tuple(z.dask.layers[z.name].indices)
    zz, = dask.optimize(z)
    z_top_after = tuple(z.dask.layers[z.name].indices)
    assert z_top_before == z_top_after, 'z_top mutated'
    dsk = optimize_blockwise(z.dask, keys=list(dask.core.flatten(z.__dask_keys__())))
    assert isinstance(dsk, HighLevelGraph)
    assert len([layer for layer in dsk.layers.values() if isinstance(layer, Blockwise)]) == 1
    dsk = optimize_blockwise(HighLevelGraph.merge(w.dask, z.dask), keys=list(dask.core.flatten([w.__dask_keys__(), z.__dask_keys__()])))
    assert isinstance(dsk, HighLevelGraph)
    assert len([layer for layer in z.dask.layers.values() if isinstance(layer, Blockwise)]) >= 1