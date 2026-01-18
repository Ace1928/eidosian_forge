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
def collections_with_node_counters():
    cnt = NodeCounter()
    df = pd.DataFrame({'x': list(range(10))})
    colls = [delayed(cnt.f)('Hello 1'), da.ones((10, 10), chunks=5).map_blocks(cnt.f), da.ones((1,), chunks=-1).map_blocks(cnt.f), db.from_sequence([1, 2], npartitions=2).map(cnt.f), db.from_sequence([1], npartitions=1).map(cnt.f), db.Item.from_delayed(delayed(cnt.f)('Hello 2')), dd.from_pandas(df, npartitions=2).map_partitions(cnt.f), dd.from_pandas(df, npartitions=1).map_partitions(cnt.f), dd.from_pandas(df['x'], npartitions=2).map_partitions(cnt.f), dd.from_pandas(df['x'], npartitions=1).map_partitions(cnt.f)]
    cnt.n = 0
    return (colls, cnt)