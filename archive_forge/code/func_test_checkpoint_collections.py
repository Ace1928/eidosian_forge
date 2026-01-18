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
def test_checkpoint_collections():
    colls, cnt = collections_with_node_counters()
    cp = checkpoint(*colls)
    cp.compute(scheduler='sync')
    assert cnt.n == 16