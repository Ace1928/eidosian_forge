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
def test__to_cytoscape_json_collapse_outputs_and_verbose():
    data = _to_cytoscape_json(dsk, collapse_outputs=True, verbose=True)
    labels = list(map(lambda x: x['data']['label'], data['nodes']))
    assert len(labels) == 6
    assert set(labels) == {'a', 'b', 'c', 'd', 'e', 'f'}
    shapes = list(map(lambda x: x['data']['shape'], data['nodes']))
    assert set(shapes) == {'ellipse', 'rectangle'}