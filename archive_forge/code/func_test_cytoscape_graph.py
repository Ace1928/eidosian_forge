from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial
from operator import add, neg
import pytest
from dask.dot import _to_cytoscape_json, cytoscape_graph
from dask import delayed
from dask.utils import ensure_not_exists
def test_cytoscape_graph(tmpdir):
    target = str(tmpdir.join('mydask.html'))
    ensure_not_exists(target)
    try:
        result = cytoscape_graph(dsk, target)
        assert os.path.isfile(target)
        assert isinstance(result, ipycytoscape.CytoscapeWidget)
    finally:
        ensure_not_exists(target)