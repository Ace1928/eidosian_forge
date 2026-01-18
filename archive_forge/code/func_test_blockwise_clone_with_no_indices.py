from __future__ import annotations
import random
import time
from operator import add
import pytest
import dask
import dask.bag as db
from dask import delayed
from dask.base import clone_key
from dask.blockwise import Blockwise
from dask.graph_manipulation import bind, checkpoint, chunks, clone, wait_on
from dask.highlevelgraph import HighLevelGraph
from dask.tests.test_base import Tuple
from dask.utils_test import import_or_none
@pytest.mark.skipif('not da or not zarr')
def test_blockwise_clone_with_no_indices():
    """https://github.com/dask/dask/issues/9621

    clone() on a Blockwise layer on top of a dependency layer with no indices
    """
    blk = da.from_zarr(zarr.ones(10))
    assert isinstance(blk.dask.layers[blk.name], Blockwise)
    assert any((isinstance(k, str) for k in blk.dask))
    cln = clone(blk)
    assert_no_common_keys(blk, cln, layers=True)
    da.utils.assert_eq(blk, cln)