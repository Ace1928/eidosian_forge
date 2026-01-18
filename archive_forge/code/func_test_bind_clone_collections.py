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
@pytest.mark.skipif('not da or not dd')
@pytest.mark.parametrize('func', [bind, clone])
def test_bind_clone_collections(func):
    if dd._dask_expr_enabled():
        pytest.skip('not supported')

    @delayed
    def double(x):
        return x * 2
    d1 = double(2)
    d2 = double(d1)
    a1 = da.ones((10, 10), chunks=5)
    a2 = a1 + 1
    a3 = a2.T
    b1 = db.from_sequence([1, 2], npartitions=2)
    b2 = b1.map(lambda x: x * 2)
    b3 = b2.map(lambda x: x + 1)
    b4 = b3.min()
    df = pd.DataFrame({'x': list(range(10))})
    ddf1 = dd.from_pandas(df, npartitions=2)
    ddf2 = ddf1.map_partitions(lambda x: x * 2)
    ddf3 = ddf2.map_partitions(lambda x: x + 1)
    ddf4 = ddf3['x']
    ddf5 = ddf4.min()
    cnt = NodeCounter()
    if func is bind:
        parent = da.ones((10, 10), chunks=5).map_blocks(cnt.f)
        cnt.n = 0
        d2c, a3c, b3c, b4c, ddf3c, ddf4c, ddf5c = bind(children=(d2, a3, b3, b4, ddf3, ddf4, ddf5), parents=parent, omit=(d1, a1, b2, ddf2), seed=0)
    else:
        d2c, a3c, b3c, b4c, ddf3c, ddf4c, ddf5c = clone(d2, a3, b3, b4, ddf3, ddf4, ddf5, omit=(d1, a1, b2, ddf2), seed=0)
    assert_did_not_materialize(d2c, d2)
    assert_did_not_materialize(a3c, a3)
    assert_did_not_materialize(b3c, b3)
    assert_did_not_materialize(b4c, b4)
    assert_did_not_materialize(ddf3c, ddf3)
    assert_did_not_materialize(ddf4c, ddf4)
    assert_did_not_materialize(ddf5c, ddf5)
    assert_no_common_keys(d2c, d2, omit=d1, layers=True)
    assert_no_common_keys(a3c, a3, omit=a1, layers=True)
    assert_no_common_keys(b3c, b3, omit=b2, layers=True)
    assert_no_common_keys(ddf3c, ddf3, omit=ddf2, layers=True)
    assert_no_common_keys(ddf4c, ddf4, omit=ddf2, layers=True)
    assert_no_common_keys(ddf5c, ddf5, omit=ddf2, layers=True)
    assert d2.compute() == d2c.compute()
    assert cnt.n == 4 or func is clone
    da.utils.assert_eq(a3c, a3)
    assert cnt.n == 8 or func is clone
    db.utils.assert_eq(b3c, b3)
    assert cnt.n == 12 or func is clone
    db.utils.assert_eq(b4c, b4)
    assert cnt.n == 16 or func is clone
    dd.utils.assert_eq(ddf3c, ddf3)
    assert cnt.n == 24 or func is clone
    dd.utils.assert_eq(ddf4c, ddf4)
    assert cnt.n == 32 or func is clone
    dd.utils.assert_eq(ddf5c, ddf5)
    assert cnt.n == 36 or func is clone