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
def test_cull_layers():
    hg = HighLevelGraph({'a': {'a1': 'd1', 'a2': 'e1'}, 'b': {'b': 'd', 'dontcull_b': 1}, 'c': {'dontcull_c': 1}, 'd': {'d': 1, 'dontcull_d': 1}, 'e': {'e': 1, 'dontcull_e': 1}}, {'a': {'d', 'e'}, 'b': {'d'}, 'c': set(), 'd': set(), 'e': set()})
    expect = HighLevelGraph({k: dict(v) for k, v in hg.layers.items() if k != 'c'}, {k: set(v) for k, v in hg.dependencies.items() if k != 'c'})
    culled = hg.cull_layers(['a', 'b'])
    assert culled.layers == expect.layers
    assert culled.dependencies == expect.dependencies
    for k in culled.layers:
        assert culled.layers[k] is hg.layers[k]
        assert culled.dependencies[k] is hg.dependencies[k]