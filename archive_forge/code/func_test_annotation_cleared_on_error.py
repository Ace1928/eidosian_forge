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
def test_annotation_cleared_on_error():
    with dask.annotate(x=1):
        with pytest.raises(ZeroDivisionError):
            with dask.annotate(x=2):
                assert dask.get_annotations() == {'x': 2}
                1 / 0
        assert dask.get_annotations() == {'x': 1}
    assert not dask.get_annotations()