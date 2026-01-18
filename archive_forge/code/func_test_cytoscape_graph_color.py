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
def test_cytoscape_graph_color():
    pytest.importorskip('matplotlib.pyplot')
    from dask.delayed import Delayed
    g = Delayed('f', dsk).visualize(engine='cytoscape')
    init_color = g.graph.nodes[0].data['color']
    g = Delayed('f', dsk).visualize(engine='cytoscape', color='order')
    assert any((n.data['color'] != init_color for n in g.graph.nodes))