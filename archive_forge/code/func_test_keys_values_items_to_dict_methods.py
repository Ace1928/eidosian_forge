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
def test_keys_values_items_to_dict_methods():
    da = pytest.importorskip('dask.array')
    a = da.ones(10, chunks=(5,))
    b = a + 1
    c = a + 2
    d = b + c
    hg = d.dask
    keys, values, items = (hg.keys(), hg.values(), hg.items())
    assert isinstance(keys, Set)
    assert list(keys) == list(hg)
    assert list(values) == [hg[i] for i in hg]
    assert list(items) == list(zip(keys, values))
    assert hg.to_dict() == dict(hg)