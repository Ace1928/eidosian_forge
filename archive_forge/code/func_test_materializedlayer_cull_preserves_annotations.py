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
def test_materializedlayer_cull_preserves_annotations():
    layer = MaterializedLayer({'a': 42, 'b': 3.14}, annotations={'foo': 'bar'})
    culled_layer, _ = layer.cull({'a'}, [])
    assert len(culled_layer) == 1
    assert culled_layer.annotations == {'foo': 'bar'}