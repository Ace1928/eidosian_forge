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
@ipython_not_installed_mark
@pytest.mark.xfail(sys.platform == 'win32', reason='graphviz/pango on conda-forge currently broken for windows', strict=False)
def test_dot_graph_defaults():
    default_name = 'mydask'
    default_format = 'png'
    target = '.'.join([default_name, default_format])
    ensure_not_exists(target)
    try:
        result = dot_graph(dsk)
        assert os.path.isfile(target)
        assert isinstance(result, Image)
    finally:
        ensure_not_exists(target)