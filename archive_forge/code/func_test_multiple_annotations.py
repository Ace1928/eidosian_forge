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
def test_multiple_annotations():
    da = pytest.importorskip('dask.array')
    with dask.annotate(block_id=annot_map_fn):
        with dask.annotate(resources={'GPU': 1}):
            A = da.ones((10, 10), chunks=(5, 5))
        B = A + 1
    C = B + 1
    assert not dask.get_annotations()
    alayer = A.__dask_graph__().layers[A.name]
    blayer = B.__dask_graph__().layers[B.name]
    clayer = C.__dask_graph__().layers[C.name]
    assert alayer.annotations == {'resources': {'GPU': 1}, 'block_id': annot_map_fn}
    assert blayer.annotations == {'block_id': annot_map_fn}
    assert clayer.annotations is None